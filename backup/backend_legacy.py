from flask import Flask, request, jsonify, Response, send_from_directory
import os
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List
import asyncio
from queue import Queue, Empty
import threading
import requests
import json

app = Flask(__name__, static_folder='.')
OLLAMA_API_BASE = "http://localhost:11434/api/generate"

# Serve index.html at root
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Serve any other static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# Custom streaming callback handler
class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue
        self.streaming_text = ""
        app.logger.info("StreamingHandler initialized")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.streaming_text += token
        app.logger.debug(f"New token received: {token}")
        # Stream each token for real-time updates
        self.queue.put(json.dumps({"token": token}))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        app.logger.info("LLM processing ended")
        app.logger.info(f"Full streaming text: {self.streaming_text}")
        
        # Only try to parse if we have content
        if not self.streaming_text.strip():
            self.queue.put(json.dumps({"error": "Empty response from LLM"}))
            return
        
        try:
            # First try to parse as structured output
            parsed_output = output_parser.parse(self.streaming_text)
            app.logger.info(f"Successfully parsed output: {parsed_output}")
            self.queue.put(json.dumps({
                "final_output": {
                    "common_points": parsed_output["common_points"],
                    "synthesized_summary": parsed_output["synthesized_summary"]
                }
            }))
        except Exception as e:
            # If parsing fails, send the raw text as a fallback
            app.logger.error(f"Parsing error: {str(e)}")
            app.logger.error(f"Failed to parse text: {self.streaming_text}")
            self.queue.put(json.dumps({
                "final_output": {
                    "common_points": ["Unable to parse structured output"],
                    "synthesized_summary": self.streaming_text
                }
            }))

# Define response schemas
response_schemas = [
    ResponseSchema(
        name="common_points",
        description="List of 3-5 key points found in both responses, each as a complete sentence"
    ),
    ResponseSchema(
        name="synthesized_summary",
        description="A concise paragraph that synthesizes the common points"
    )
]

# Create output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create comparison prompt template
COMPARE_TEMPLATE = """
You are a precise and analytical AI assistant. Your task is to compare two AI responses and synthesize their common ground.

Question asked: {question}

Response 1:
{response1}

Response 2:
{response2}

Instructions:
1. Identify 3-5 key points that are present in both responses
2. Synthesize these common points into a coherent summary paragraph
3. Ignore any contradicting or unique points
4. Keep the response concise and focused

Your response MUST follow this exact format:
{format_instructions}

Remember to structure your response exactly as specified above. Begin your analysis now:
"""

# Initialize prompt template
prompt = PromptTemplate(
    template=COMPARE_TEMPLATE,
    input_variables=["question", "response1", "response2"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

@app.route("/get_summary", methods=["POST"])
def get_summary():
    app.logger.info("=====================================")
    app.logger.info("SUMMARY ENDPOINT CALLED")
    app.logger.info("=====================================")
    
    try:
        data = request.json
        if not data:
            app.logger.error("No data provided to summary endpoint")
            return jsonify({"error": "No data provided"}), 400

        question = data.get("question", "")
        response1 = data.get("response1", "")
        response2 = data.get("response2", "")

        app.logger.info(f"Received summary request with question: {question}")
        app.logger.info(f"Response1 length: {len(response1)}")
        app.logger.info(f"Response2 length: {len(response2)}")

        if not response1 or not response2:
            return jsonify({"error": "Missing responses"}), 400

        # Create the prompt
        prompt = f"""Analyze these two AI responses and find their common points:

QUESTION: {question}

FIRST RESPONSE:
{response1}

SECOND RESPONSE:
{response2}

Provide your analysis in this format:
Key Common Points:
- [Point 1]
- [Point 2]
- [Point 3]

Summary:
[Write a brief summary paragraph]

Begin analysis:"""

        def generate():
            try:
                # Use the same approach that works in get_response
                response = requests.post(
                    OLLAMA_API_BASE,
                    json={
                        "model": "llama3.1:8b-text-q6_K",
                        "prompt": prompt,
                        "stream": True
                    },
                    stream=True
                )

                accumulated_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            if 'response' in json_response:
                                token = json_response['response']
                                accumulated_response += token
                                yield f"data: {json.dumps({'token': token})}\n\n"

                        except json.JSONDecodeError:
                            continue

                # After all tokens are received, parse the complete response
                try:
                    # Split into sections
                    sections = accumulated_response.split('\n\n')
                    points_section = next((s for s in sections if 'Key Common Points:' in s), '')
                    summary_section = next((s for s in sections if 'Summary:' in s), '')

                    # Extract points
                    points = [
                        p.strip('- ').strip()
                        for p in points_section.split('\n')
                        if p.strip().startswith('-')
                    ]

                    # Extract summary
                    summary = summary_section.replace('Summary:', '').strip()

                    final_output = {
                        "common_points": [p for p in points if len(p) > 10][:5],
                        "synthesized_summary": summary if summary else "Summary not found in response"
                    }

                    yield f"data: {json.dumps({'final_output': final_output})}\n\n"

                except Exception as e:
                    app.logger.error(f"Error parsing final response: {str(e)}")
                    error_output = {
                        'final_output': {
                            'common_points': ['Parsing error - sending raw response'],
                            'synthesized_summary': accumulated_response
                        }
                    }
                    yield f"data: {json.dumps(error_output)}\n\n"

            except Exception as e:
                app.logger.error(f"Error in generate: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        app.logger.error(f"Unexpected error in get_summary: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add this new route to get available models
@app.route('/get_models', methods=['GET'])
def get_models():
    try:
        # Call Ollama API to get available models
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            data = response.json()
            # Extract model names from the response
            if 'models' in data and isinstance(data['models'], list):
                model_names = [model['name'] for model in data['models']]
            else:
                # Fallback for different API response format
                model_names = list(data.keys()) if isinstance(data, dict) else []
            
            app.logger.info(f"Available models: {model_names}")
            return jsonify({"models": model_names})
        else:
            app.logger.error(f"Failed to fetch models: {response.status_code}")
            return jsonify({"error": "Failed to fetch models from Ollama"}), 500
    except Exception as e:
        app.logger.error(f"Error fetching models: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add favicon route to handle the 404
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return no content

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data = request.json
        TOKEN_LIMIT = 250

        # Add system prompt for concise responses
        SYSTEM_PROMPT = """Please provide clear and concise responses around 200 tokens. 
        Focus on the most important information and be direct in your answers. 
        If a longer response is needed, prioritize the most crucial points first.
        In no chance shall an answer be longer than 250 tokens."""

        # Combine system prompt with user prompt
        combined_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {data['prompt']}"
        
        def generate():
            try:
                responses = {"response1": "", "response2": ""}
                tokens = {"response1": 0, "response2": 0}

                # Model 1 response with system prompt
                response1 = requests.post(
                    OLLAMA_API_BASE,
                    json={
                        "model": data['model1'],
                        "prompt": combined_prompt,
                        "system": SYSTEM_PROMPT,  # Add system prompt to Ollama context
                        "stream": True
                    },
                    stream=True
                )

                for line in response1.iter_lines():
                    if line and tokens["response1"] < TOKEN_LIMIT:
                        try:
                            json_response = json.loads(line)
                            response_text = json_response.get('response', '')
                            if response_text:
                                responses["response1"] += response_text
                                tokens["response1"] += 1
                                yield f"data: {json.dumps({'response1': response_text, 'tokens1': tokens['response1']})}\n\n"
                                
                                if tokens["response1"] >= TOKEN_LIMIT:
                                    message = json.dumps({'response1': '\n[Token limit reached: 250]', 'tokens1': tokens['response1']})
                                    yield f"data: {message}\n\n"
                                    break
                        except json.JSONDecodeError:
                            continue

                # Model 2 response with system prompt
                response2 = requests.post(
                    OLLAMA_API_BASE,
                    json={
                        "model": data['model2'],
                        "prompt": combined_prompt,
                        "system": SYSTEM_PROMPT,  # Add system prompt to Ollama context
                        "stream": True
                    },
                    stream=True
                )

                for line in response2.iter_lines():
                    if line and tokens["response2"] < TOKEN_LIMIT:
                        try:
                            json_response = json.loads(line)
                            response_text = json_response.get('response', '')
                            if response_text:
                                responses["response2"] += response_text
                                tokens["response2"] += 1
                                yield f"data: {json.dumps({'response2': response_text, 'tokens2': tokens['response2']})}\n\n"
                                
                                if tokens["response2"] >= TOKEN_LIMIT:
                                    message = json.dumps({'response2': '\n[Token limit reached: 250]', 'tokens2': tokens['response2']})
                                    yield f"data: {message}\n\n"
                                    break
                        except json.JSONDecodeError:
                            continue

                yield f"data: {json.dumps({'done': True, 'final_tokens': tokens})}\n\n"

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                app.logger.error(error_msg)
                yield f"data: {json.dumps({'error': error_msg})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Keep existing routes and functions
# ... rest of your code ...

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)