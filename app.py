from flask import Flask, request, jsonify
from flask_cors import CORS
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Set, Dict, Any
import os
import random
import json
from urllib.parse import unquote
from threading import Thread
import re
import asyncio
from functools import wraps
import firebase_admin
from firebase_admin import credentials, firestore
from langchain_community.llms import OpenAI, Anthropic
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from google.cloud.firestore_v1 import AsyncClient
import time
from copy import deepcopy

app = Flask(__name__)
CORS(app)
load_dotenv()

# Environment variables for different API keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Initialize Firebase Admin SDK with your credentials
try:
    cred = credentials.Certificate(r'D:\bck\school-6ab22-firebase-adminsdk-fbsvc-d4466910bb.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firestore initialized successfully")
except Exception as e:
    print(f"Error initializing Firestore: {str(e)}")
    db = None

# Constants
TEST_MODE_QUESTIONS = 20  # Default value
TEST_MODE_SET_SIZE = 10  # Default questions per set in test mode
TEST_MODE_SETS = 2  # Default total number of sets
PRACTICE_MODE_QUESTIONS_PER_SET = 5
HARD_QUESTION_PERCENTAGE = 70

# Cache for model instances
model_cache = {}

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str
    explanation: str

# Global variables
question_cache: List[QuizQuestion] = []
used_questions: Set[str] = set()
current_topic: str = ""
file_content_cache: Dict[str, str] = {}
concept_cache: Dict[str, str] = {}
processed_files: Set[str] = set()

def async_to_sync(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

async def get_model_config() -> Dict[str, Any]:
    """Fetch model configuration from Firestore"""
    if db is None:
        print("Firestore not initialized, using default model (Gemini)")
        return {
            "provider": "google",
            "name": "gemini-2.0-flash-exp",
            "description": "Google Gemini 2.0 Flash"
        }
        
    try:
        # Get the active model from Firestore
        active_model_ref = db.collection('config').document('active_model').get()
        if not active_model_ref.exists:
            print("No active model document found, using default model (Gemini)")
            return {
                "provider": "google",
                "name": "gemini-2.0-flash-exp",
                "description": "Google Gemini 2.0 Flash"
            }
            
        active_model_data = active_model_ref.to_dict()
        model_provider = active_model_data.get('provider', 'google')
        model_name = active_model_data.get('model', 'gemini-2.0-flash-exp')
        
        print(f"Active model from config: provider={model_provider}, model={model_name}")
        
        # Get the detailed model configuration
        model_ref = db.collection('models').document(model_provider).collection('models').document(model_name).get()
        if not model_ref.exists:
            print(f"Model config not found for {model_provider}/{model_name}, using default")
            return {
                "provider": model_provider,
                "name": model_name,
                "description": f"{model_provider} {model_name}"
            }
            
        model_config = model_ref.to_dict()
        print(f"Model config from Firestore: {model_config}")
        
        # Make sure we're returning the provider from active_model_data
        final_config = {
            "provider": model_provider,  # Use the provider from active_model_data
            "name": model_name,
            "description": model_config.get('description', f"{model_provider}/{model_name}")
        }
        
        print(f"Final model config: {final_config}")
        return final_config
        
    except Exception as e:
        print(f"Error fetching model config: {str(e)}")
        return {
            "provider": "google",
            "name": "gemini-2.0-flash-exp",
            "description": "Google Gemini 2.0 Flash (Default)"
        }
async def save_questions_to_firestore(questions, standard, subject, chapter, is_practice_mode, prompt, model_config=None):
    if db is None:
        print("Firestore not initialized, cannot save questions")
        return False
    
    try:
        # Get model info for metadata
        if model_config is None:
            model_config = await get_model_config()
        
        provider = model_config.get("provider", "google")
        model_name = model_config.get("name", "gemini-2.0-flash-exp")
        
        # Create a timestamp for the document ID
        timestamp = int(time.time() * 1000)
        
        # Determine mode for path construction
        mode = "practice_results" if is_practice_mode else "test_results"
        
        # Construct the Firestore path for the main document
        doc_path = f"Quiz_questions/{standard}/subjects/{subject}/chapters/{chapter}/{mode}/{timestamp}"
        
        print(f"Saving {len(questions)} questions to: {doc_path}")
        
        # Extract only the base prompt template, not including the content or questions
        # First try to extract just the general prompt template
        base_prompt = prompt
        if "Using this content:" in prompt:
            base_prompt = prompt.split("Using this content:")[0].strip()
        elif "Generate questions about" in prompt:
            base_prompt = prompt.split("Generate questions about")[0].strip()
        elif "Generate " in prompt and " questions " in prompt:
            # More generic approach for various prompt formats
            match = re.search(r'(.*?)Generate\s+\d+\s+questions', prompt, re.DOTALL)
            if match:
                base_prompt = match.group(1).strip()
        
        print(f"Storing base prompt template (length: {len(base_prompt)})")
        
        # Create a document with metadata
        doc_ref = db.document(doc_path)
        doc_ref.set({
            "timestamp": timestamp,
            "model_provider": provider,
            "model_name": model_name,
            "prompt": base_prompt,  # Only store the prompt template
            "standard": standard,
            "subject": subject,
            "chapter": chapter,
            "mode": "practice" if is_practice_mode else "test",
            "question_count": len(questions)
        })
        
        # Add each question as a document in a subcollection
        # Process in smaller batches to avoid Firestore limits
        batch_size = 200
        for i in range(0, len(questions), batch_size):
            batch = db.batch()
            
            for j, question in enumerate(questions[i:i+batch_size]):
                # Store in the format specified
                question_data = {
                    'questionText': question.question,
                    'options': question.options,
                    'correctAnswer': question.answer,
                    'explanation': question.explanation,
                    'isCorrect': False,  # Default value as no user answer yet
                    'questionNumber': i + j + 1,
                }
                
                question_ref = doc_ref.collection("questions").document(f"question_{i+j}")
                batch.set(question_ref, question_data)
            
            # Commit the batch
            batch.commit()
            print(f"Saved batch of questions {i} to {i+min(batch_size, len(questions)-i)} to Firestore")
        
        print(f"Successfully saved all {len(questions)} questions to Firestore")
        return True
    
    except Exception as e:
        print(f"Error saving questions to Firestore: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
async def get_llm_client(model_config=None):
    """Get the appropriate LLM client based on configuration"""
    if model_config is None:
        model_config = await get_model_config()
        
    provider = model_config.get("provider", "google").lower()
    model_name = model_config.get("name", "gemini-2.0-flash-exp")
    
    print(f"Creating LLM client for provider: {provider}, model: {model_name}")
    
    # Create a cache key for this model
    cache_key = f"{provider}_{model_name}"
    
    # Check if we already have this model in cache
    if cache_key in model_cache:
        print(f"Using cached model: {cache_key}")
        return model_cache[cache_key]
    try:
        if provider == "google":
            # Initialize Google Generative AI model
            if not GEMINI_API_KEY:
                raise ValueError("Google API key not found in environment")
            genai.configure(api_key=GEMINI_API_KEY)
            model = ChatGoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY, temperature=0.7)
            
        elif provider == "openai":
            # Initialize OpenAI model
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found in environment")
            model = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=model_name, temperature=0.7)
            
        elif provider == "anthropic":
            # Initialize Anthropic model
            if not ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found in environment")
            model = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model_name=model_name, temperature=0.7)
            
        else:
            print(f"Unknown provider: {provider}, defaulting to Google")
            if not GEMINI_API_KEY:
                raise ValueError("Google API key not found in environment")
            genai.configure(api_key=GEMINI_API_KEY)
            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY, temperature=0.7)
        
        # Cache the model instance
        model_cache[cache_key] = model
        return model
        
    except Exception as e:
        print(f"Error initializing model {provider}/{model_name}: {str(e)}")
        print("Falling back to default Google model")
        if not GEMINI_API_KEY:
            raise ValueError("Google API key not found in environment")
        genai.configure(api_key=GEMINI_API_KEY)
        fallback_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY, temperature=0.7)
        model_cache["fallback"] = fallback_model
        return fallback_model
async def generate_content(prompt, model_config=None):
    """Generate content using the configured LLM"""
    model = await get_llm_client(model_config)
    
    try:
        # Format messages based on LangChain expectations
        messages = [HumanMessage(content=prompt)]
        
        # Generate content
        response = await asyncio.to_thread(lambda: model.invoke(messages))
        
        # Extract text from the response based on model type
        if isinstance(response, AIMessage):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        elif isinstance(response, str):
            return response
        else:
            return str(response)
            
    except Exception as e:
        print(f"Error generating content: {str(e)}")
        raise

async def get_custom_prompt(standard: str, subject: str, chapter: str, is_practice_mode: bool) -> Optional[dict]:
    if db is None:
        print("Firestore not initialized, using default prompt")
        return None
        
    try:
        # Construct the Firestore path based on the parameters
        mode = "Practice mode" if is_practice_mode else "Test mode"
        path = f"Tuning/Standards/{standard}/Subject/{subject}/{mode}/Chapters/{chapter}"
        print(f"Attempting to fetch custom prompt from path: {path}")
        
        # First try with the standard path
        doc_ref = db.collection('Tuning').document('Standards').collection(standard).document('Subject') \
                    .collection(subject).document(mode).collection('Chapters').document(chapter)
        
        doc = doc_ref.get()
        
        # If document not found, try searching for it with a case-insensitive approach
        if not doc.exists:
            print(f"Document not found at exact path, trying to find matching documents...")
            # Get all chapters in the collection
            chapters_ref = db.collection('Tuning').document('Standards').collection(standard).document('Subject') \
                            .collection(subject).document(mode).collection('Chapters')
            chapters = chapters_ref.get()
            
            # Check if any document ID matches case-insensitively
            for chapter_doc in chapters:
                if chapter_doc.id.lower() == chapter.lower():
                    print(f"Found matching document with ID: {chapter_doc.id}")
                    doc = chapter_doc.reference.get()
                    break
        
        if doc.exists:
            prompt_data = doc.to_dict() or {}
            result = {}
            
            # Extract prompt text
            for field_name in ['prompt', 'Prompt', 'PROMPT']:
                if field_name in prompt_data:
                    print(f"Custom prompt found in Firestore (field: {field_name}) for {standard}/{subject}/{chapter}/{mode}")
                    result['prompt_text'] = prompt_data[field_name]
                    break
            
            # Extract test parameters (only for test mode)
            if not is_practice_mode:
                # Extract total_questions parameter
                for field_name in ['total_questions', 'totalQuestions', 'TOTAL_QUESTIONS']:
                    if field_name in prompt_data:
                        try:
                            result['total_questions'] = int(prompt_data[field_name])
                            print(f"Found total_questions: {result['total_questions']}")
                            break
                        except (ValueError, TypeError):
                            print(f"Invalid total_questions value: {prompt_data[field_name]}")
                
                # Extract number_of_sets parameter
                for field_name in ['number_of_sets', 'numberOfSets', 'NUMBER_OF_SETS']:
                    if field_name in prompt_data:
                        try:
                            result['number_of_sets'] = int(prompt_data[field_name])
                            print(f"Found number_of_sets: {result['number_of_sets']}")
                            break
                        except (ValueError, TypeError):
                            print(f"Invalid number_of_sets value: {prompt_data[field_name]}")
                
                # Extract time_in_minutes parameter
                for field_name in ['time_in_minutes', 'timeInMinutes', 'TIME_IN_MINUTES']:
                    if field_name in prompt_data:
                        try:
                            result['time_in_minutes'] = int(prompt_data[field_name])
                            print(f"Found time_in_minutes: {result['time_in_minutes']}")
                            break
                        except (ValueError, TypeError):
                            print(f"Invalid time_in_minutes value: {prompt_data[field_name]}")
            
            if 'prompt_text' in result:
                return result
            else:
                print(f"Document exists but no prompt field found for {standard}/{subject}/{chapter}/{mode}")
                print(f"Available fields: {list(prompt_data.keys())}")
                return None
        else:
            print(f"No document found in Firestore for {standard}/{subject}/{chapter}/{mode}")
            
            # Debugging: list all documents in the chapters collection
            try:
                chapters_ref = db.collection('Tuning').document('Standards').collection(standard).document('Subject') \
                              .collection(subject).document(mode).collection('Chapters')
                chapters = chapters_ref.get()
                print(f"Available chapters in {standard}/{subject}/{mode}:")
                for chapter_doc in chapters:
                    print(f"  - {chapter_doc.id}")
            except Exception as e:
                print(f"Error listing chapters: {str(e)}")
            
            return None
    except Exception as e:
        print(f"Error fetching custom prompt from Firestore: {str(e)}")
        return None
    
async def extract_key_concepts(text_content: str) -> str:
    print("\nExtracting key concepts from text content...")
    prompt = """Extract and summarize the key concepts and important points from this text. 
    Include only the most essential information needed for generating questions later.
    Keep it concise but comprehensive."""
    
    try:
        concepts = await generate_content(f"{prompt}\n\nText: {text_content}")
        print("Successfully extracted key concepts")
        return concepts
    except Exception as e:
        print(f"Error extracting key concepts: {str(e)}")
        # Return a subset of the text as fallback
        return text_content[:1000] + "..."

def print_question(q: QuizQuestion, index: int):
    print(f"\nQuestion {index}:")
    print("=" * 50)
    print(f"Q: {q.question}")
    print("\nOptions:")
    for i, opt in enumerate(q.options):
        print(f"{chr(65+i)}) {opt}")
    print(f"\nCorrect Answer: {q.answer}")
    print(f"Explanation: {q.explanation}")
    print("-" * 50)

def read_and_process_content(file_path: str) -> Optional[str]:
    try:
        print(f"\nReading and processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Cache the full content for later use
            file_content_cache[file_path] = content
            print("Successfully processed file")
            return content
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

async def process_and_extract_concepts(file_path: str) -> Optional[str]:
    """Process a file and extract key concepts, with caching"""
    # Check if we already have the concepts in cache
    if file_path in concept_cache:
        print(f"Using cached concepts for {file_path}")
        return concept_cache[file_path]
        
    # Read the file content (or use cache)
    if file_path in file_content_cache:
        content = file_content_cache[file_path]
    else:
        content = read_and_process_content(file_path)
        if not content:
            return None
    
    # Extract key concepts
    key_concepts = await extract_key_concepts(content)
    concept_cache[file_path] = key_concepts
    return key_concepts

def calculate_accuracy(text_content: str, questions: List[QuizQuestion]) -> float:
    try:
        total_words = len(text_content.split())
        relevant_count = 0
        for q in questions:
            question_words = q.question.lower().split()
            for word in question_words:
                if len(word) > 3 and word in text_content.lower():
                    relevant_count += 1
        accuracy = min((relevant_count / (len(questions) * 2)) * 100, 100)
        return round(accuracy, 2)
    except Exception as e:
        print(f"Error calculating accuracy: {str(e)}")
        return 0.0

async def generate_quiz_questions(text_content: str = None, topic: str = None, concepts: str = None, 
                               is_practice_mode: bool = True, custom_prompt: str = None,
                               standard: str = None, subject: str = None, 
                               num_questions: int = None) -> Optional[List[QuizQuestion]]:
    print("\nGenerating Questions...")
    print("=" * 50)
    print(f"Mode: {'Practice' if is_practice_mode else 'Test'}")
    print(f"Source: {'Text File' if text_content else 'Concepts' if concepts else 'Topic only'}")
    print(f"Custom Prompt: {'Yes' if custom_prompt else 'No (using default)'}")
    print(f"Requested number of questions: {num_questions}")

    try:
        # Use provided num_questions if specified, otherwise use defaults
        if num_questions is None:
            num_questions = PRACTICE_MODE_QUESTIONS_PER_SET if is_practice_mode else TEST_MODE_SET_SIZE
        
        print(f"Generating {num_questions} questions")

        # Use custom prompt if available, otherwise use default
        if custom_prompt:
            enhanced_prompt = custom_prompt
            print("Using custom prompt from Firestore")
        else:
            if not is_practice_mode:
                enhanced_prompt = """Generate extremely challenging multiple choice questions that test advanced cognitive abilities. Questions should be:

                Question Distribution:
                1. Complex logical reasoning (40%)
                   - Multi-step deductive reasoning
                   - Advanced pattern recognition
                   - Abstract concept application
                
                2. Advanced critical thinking (60%)
                   - Deep analysis requirements
                   - Complex problem evaluation
                   - Multi-perspective consideration
                
                3. Multi-step problem solving (60%)
                   - Sophisticated computational thinking
                   - Strategic solution planning
                   - Advanced concept integration

                Requirements:
                - All questions must be at the highest difficulty level
                - Questions should challenge even advanced learners
                - Clear and unambiguous despite complexity
                - Each question should require deep understanding
                - Include detailed explanations for learning

                Format Requirements:
                - 4 distinct options per question
                - One definitively correct answer
                - Comprehensive explanation (40 words)
                - Crystal clear question structure

                Response Format (JSON):
                {
                    "questions": [
                        {
                            "question": "Question text",
                            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                            "answer": "Correct option text",
                            "explanation": "Detailed explanation"
                        }
                    ]
                }"""
            else:
                enhanced_prompt = """Generate a balanced mix of multiple choice questions with varying difficulty levels:

                Question Distribution:
                1. Hard questions (100%)
                   - Complex reasoning
                   - Advanced problem-solving
                   - Deep conceptual understanding
                
                2. Intermediate questions (100%)
                   - Applied knowledge
                   - Basic analysis
                   - Concept integration
                
                3. Basic questions (1000%)
                   - Fundamental concepts
                   - Direct application
                   - Core understanding

                Requirements:
                - Progressive difficulty level
                - Clear learning progression
                - Balanced concept coverage
                - Appropriate challenge level
                - Helpful explanations for learning

                Format Requirements:
                - 4 distinct options per question
                - One definitively correct answer
                - Clear explanation (40 words)
                - Well-structured questions

                Response Format (JSON):
                {
                    "questions": [
                        {
                            "question": "Question text",
                            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                            "answer": "Correct option text",
                            "explanation": "Detailed explanation"
                        }
                    ]
                }"""
            print("Using default prompt")

        if text_content:
            content_prompt = f"Using this content:\n{text_content}\n\nGenerate {num_questions} questions that follow the guidelines above."
            print("Using full text content for generation")
        elif concepts:
            content_prompt = f"""Generate {num_questions} questions about {topic} using these key concepts:
            {concepts}
            
            Ensure questions are based on these concepts while maintaining variety and appropriate difficulty."""
            print("Using extracted concepts for generation")
        else:
            content_prompt = f"Generate {num_questions} questions about {topic} that follow the guidelines above."
            print("Using topic only for generation")

        full_prompt = f"{enhanced_prompt}\n\n{content_prompt}"

        # Get the LLM model config and generate content
        model_config = await get_model_config()
        response_text = await generate_content(full_prompt, model_config)
        response_text = response_text.strip()
        
        json_text = re.search(r'({[\s\S]*})', response_text)
        if not json_text:
            raise ValueError("No valid JSON found in response")
            
        cleaned_json = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_text.group(1))
        cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
        
        try:
            response_data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print(f"Problematic JSON: {cleaned_json}")
            raise
        
        processed_questions = []
        for q in response_data.get("questions", []):
            if not all(k in q for k in ["question", "options", "answer", "explanation"]):
                continue
                
            if len(q["options"]) != 4:
                continue
                
            if q["question"] in used_questions:
                continue

            cleaned_question = {
                "question": q["question"].strip(),
                "options": [opt.strip() for opt in q["options"]],
                "answer": q["answer"].strip(),
                "explanation": q["explanation"].strip()
            }

            try:
                question = QuizQuestion(**cleaned_question)

                if question.answer not in question.options:
                    continue

                random.shuffle(question.options)
                used_questions.add(question.question)
                processed_questions.append(question)
                print_question(question, len(processed_questions))
            except Exception as e:
                print(f"Error creating question object: {str(e)}")
                continue
            
        print(f"\nSuccessfully processed {len(processed_questions)} questions")
        
        # Save questions to Firestore (only if we have a chapter/topic, standard, and subject)
        if processed_questions and topic and standard and subject:
            # Create a background task to save the questions
            asyncio.create_task(save_questions_to_firestore(
                processed_questions, 
                standard, 
                subject, 
                topic,
                is_practice_mode,
                full_prompt,
                model_config
            ))
            
        return processed_questions

    except Exception as e:
        print(f"Error in generate_quiz_questions: {str(e)}")
        return None
    
@app.route('/quiz/prepare-next-batch', methods=['GET'])
@async_to_sync
async def prepare_next_batch():
    try:
        topic = unquote(request.args.get('topic', ''))
        standard = request.args.get('standard', '')
        subject = request.args.get('subject', '')
        is_practice_mode = request.args.get('is_practice_mode', 'true').lower() == 'true'

        if not topic or not is_practice_mode:
            return jsonify({"error": "Invalid parameters"}), 400

        # Get custom prompt from Firestore
        custom_prompt_data = await get_custom_prompt(standard, subject, topic, is_practice_mode)
        custom_prompt_text = custom_prompt_data.get('prompt_text') if custom_prompt_data else None

        # Generate new questions in background if cache is running low
        file_path = rf"D:\bck\schoolbooks\{standard}\{subject}\{topic}.txt"
        
        if os.path.exists(file_path):
            if file_path not in processed_files:
                print("\nProcessing new file content for next batch...")
                chapter_content = read_and_process_content(file_path)
                concepts = await extract_key_concepts(chapter_content)
                concept_cache[file_path] = concepts
                processed_files.add(file_path)
                new_questions = await generate_quiz_questions(
                    text_content=chapter_content,
                    is_practice_mode=True,
                    custom_prompt=custom_prompt_text,
                    standard=standard,
                    subject=subject
                )
            else:
                print("\nUsing cached concepts for next batch...")
                concepts = concept_cache.get(file_path)
                new_questions = await generate_quiz_questions(
                    topic=topic,
                    concepts=concepts,
                    is_practice_mode=True,
                    custom_prompt=custom_prompt_text,
                    standard=standard,
                    subject=subject
                )
        else:
            print("\nGenerating next batch questions from topic only...")
            new_questions = await generate_quiz_questions(
                topic=topic,
                is_practice_mode=True,
                custom_prompt=custom_prompt_text,
                standard=standard,
                subject=subject
            )
            
        if new_questions:
            question_cache.extend(new_questions)
            return jsonify({"status": "success", "message": "Next batch prepared"}), 200
        else:
            return jsonify({"error": "Failed to generate next batch"}), 500

    except Exception as e:
        print(f"Error in prepare_next_batch: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/quiz/next', methods=['GET'])
@async_to_sync
async def get_next_questions():
    try:
        topic = unquote(request.args.get('topic', ''))
        current_index = int(request.args.get('current_index', 0))
        standard = request.args.get('standard', '')
        subject = request.args.get('subject', '')
        is_practice_mode = request.args.get('is_practice_mode', 'true').lower() == 'true'
        
        if not topic:
            return jsonify({"error": "Missing topic parameter"}), 400

        topic = topic.strip()
        
        # Get custom prompt from Firestore
        custom_prompt_data = await get_custom_prompt(standard, subject, topic, is_practice_mode)
        custom_prompt_text = custom_prompt_data.get('prompt_text') if custom_prompt_data else None
        
        # Define file_path here so it's available for both modes
        file_path = rf"D:\bck\schoolbooks\{standard}\{subject}\{topic}.txt"
        
        # For test mode, use dynamic parameters if available
        if not is_practice_mode:
            # Initialize variables with default values
            total_questions = TEST_MODE_QUESTIONS
            number_of_sets = TEST_MODE_SETS
            time_in_minutes = 30
            
            # Override with custom values from Firestore if available
            if custom_prompt_data:
                if 'total_questions' in custom_prompt_data:
                    total_questions = custom_prompt_data['total_questions']
                if 'number_of_sets' in custom_prompt_data:
                    number_of_sets = custom_prompt_data['number_of_sets']
                if 'time_in_minutes' in custom_prompt_data:
                    time_in_minutes = custom_prompt_data['time_in_minutes']
            
            # Calculate questions per set
            questions_per_set = total_questions // number_of_sets
            
            # End conditions
            if current_index >= total_questions:
                return jsonify({
                    "questions": [],
                    "should_fetch": False,
                    "total_questions": total_questions,
                    "time_in_minutes": time_in_minutes
                })
            
            # Calculate which set we're on (1-based)
            current_set = (current_index // questions_per_set) + 1
            
            # Calculate if we're at the first question of a set
            is_first_question_of_set = (current_index % questions_per_set) == 0
            
            # Calculate remaining questions
            remaining_questions = total_questions - current_index
            
            # Questions we'll send to the frontend for this request
            questions_to_send = min(questions_per_set, remaining_questions)
            
            print(f"TEST MODE CONFIG: total_questions={total_questions}, number_of_sets={number_of_sets}, questions_per_set={questions_per_set}")
            print(f"Current index: {current_index}, Current set: {current_set}/{number_of_sets}")
            print(f"Remaining questions: {remaining_questions}, Will send: {questions_to_send}")
            print(f"Is first question of set: {is_first_question_of_set}")
            
            # Initialize content source variables
            if os.path.exists(file_path):
                if file_path not in processed_files:
                    print("\nProcessing new file content...")
                    content_source = read_and_process_content(file_path)
                    concepts = await extract_key_concepts(content_source)
                    concept_cache[file_path] = concepts
                    processed_files.add(file_path)
                    source_type = "text_content"
                else:
                    print("\nUsing cached concepts...")
                    content_source = concept_cache.get(file_path)
                    source_type = "concepts"
            else:
                print("\nGenerating questions from topic only...")
                content_source = topic
                source_type = "topic"
            
            # Generate questions for current set if not already in cache
            if len(question_cache) < questions_to_send:
                print(f"Generating questions for set {current_set}")
                
                # Generate questions based on source type
                if source_type == "text_content":
                    questions = await generate_quiz_questions(
                        text_content=content_source,
                        topic=topic,
                        is_practice_mode=False,
                        custom_prompt=custom_prompt_text,
                        standard=standard,
                        subject=subject,
                        num_questions=questions_to_send
                    )
                elif source_type == "concepts":
                    questions = await generate_quiz_questions(
                        concepts=content_source,
                        topic=topic,
                        is_practice_mode=False,
                        custom_prompt=custom_prompt_text,
                        standard=standard,
                        subject=subject,
                        num_questions=questions_to_send
                    )
                else:  # topic only
                    questions = await generate_quiz_questions(
                        topic=topic,
                        is_practice_mode=False,
                        custom_prompt=custom_prompt_text,
                        standard=standard,
                        subject=subject,
                        num_questions=questions_to_send
                    )
                
                # Check for errors in question generation
                if not questions or len(questions) == 0:
                    return jsonify({"error": f"Failed to generate questions for set {current_set}"}), 500
                
                print(f"Successfully generated {len(questions)} questions for set {current_set}")
                question_cache.extend(questions)
            
            # Get questions to return from cache
            questions_to_return = question_cache[:questions_to_send]
            
            # Remove sent questions from cache
            del question_cache[:questions_to_send]
            
            # Calculate if more questions should be fetched
            next_index = current_index + questions_to_send
            should_fetch_more = next_index < total_questions
            
            print(f"Next fetch should be at index: {next_index}")
            print(f"Should fetch more? {should_fetch_more}")
            
            # Return the response with all necessary information
            return jsonify({
               "questions": [q.model_dump() for q in questions_to_return],
               "should_fetch": should_fetch_more,
               "total_questions": total_questions,
               "time_in_minutes": time_in_minutes,
               "next_index": next_index,
               "current_set": current_set,
               "total_sets": number_of_sets,
               "is_last_set": current_set >= number_of_sets
           })
           
       # Practice mode code
        else:
           if current_index == 1 or len(question_cache) < PRACTICE_MODE_QUESTIONS_PER_SET:
               if os.path.exists(file_path):
                   if file_path not in processed_files:
                       print("\nProcessing new file content...")
                       chapter_content = read_and_process_content(file_path)
                       concepts = await extract_key_concepts(chapter_content)
                       concept_cache[file_path] = concepts
                       processed_files.add(file_path)
                       questions = await generate_quiz_questions(
                           text_content=chapter_content,
                           is_practice_mode=True,
                           custom_prompt=custom_prompt_text,
                           standard=standard,
                           subject=subject
                       )
                   else:
                       print("\nUsing cached concepts...")
                       concepts = concept_cache.get(file_path)
                       questions = await generate_quiz_questions(
                           topic=topic,
                           concepts=concepts,
                           is_practice_mode=True,
                           custom_prompt=custom_prompt_text,
                           standard=standard,
                           subject=subject
                       )
               else:
                   print("\nGenerating questions from topic only...")
                   questions = await generate_quiz_questions(
                       topic=topic,
                       is_practice_mode=True,
                       custom_prompt=custom_prompt_text,
                       standard=standard,
                       subject=subject
                   )
                   
               if questions is None or len(questions) == 0:
                   return jsonify({"error": "Failed to generate questions"}), 500
                   
               question_cache.extend(questions)

           questions_per_set = PRACTICE_MODE_QUESTIONS_PER_SET
           questions_to_send = question_cache[:questions_per_set]
           del question_cache[:len(questions_to_send)]

           if not questions_to_send:
               return jsonify({"error": "No questions available"}), 500

           return jsonify({
               "questions": [q.model_dump() for q in questions_to_send],
               "should_fetch": True,
               "total_questions": -1
           })
    except Exception as e:
       print(f"Error in get_next_questions: {str(e)}")
       import traceback
       traceback.print_exc()
       return jsonify({"error": str(e)}), 500
   
@app.route('/quiz/debug-info', methods=['GET'])
def debug_info():
   return jsonify({
       "cache_size": len(question_cache),
       "used_questions_count": len(used_questions),
       "test_mode_defaults": {
           "total_questions": TEST_MODE_QUESTIONS,
           "set_size": TEST_MODE_SET_SIZE,
           "sets": TEST_MODE_SETS,
       },
       "practice_mode_defaults": {
           "questions_per_set": PRACTICE_MODE_QUESTIONS_PER_SET,
       },
       "model_cache_size": len(model_cache)
   })

@app.route('/health', methods=['GET'])
def health_check():
   return jsonify({"status": "healthy"}), 200

@app.route('/quiz/clear-cache', methods=['GET'])
def clear_cache():
   global question_cache, used_questions, file_content_cache, concept_cache, processed_files
   question_cache.clear()
   used_questions.clear()
   file_content_cache.clear()
   concept_cache.clear()
   processed_files.clear()
   model_cache.clear()  # Also clear model cache
   print("\nAll caches cleared")
   return jsonify({"status": "Caches cleared"}), 200

@app.route('/quiz/status', methods=['GET'])
@async_to_sync
async def get_status():
   try:
       # Get current model configuration
       model_config = await get_model_config()
       
       return jsonify({
           "cache_size": len(question_cache),
           "used_questions": len(used_questions),
           "file_cache_size": len(file_content_cache),
           "concept_cache_size": len(concept_cache),
           "processed_files": len(processed_files),
           "current_topic": current_topic,
           "firestore_connected": db is not None,
           "current_model": {
               "provider": model_config.get("provider", "unknown"),
               "name": model_config.get("name", "unknown"),
               "description": model_config.get("description", "Unknown model")
           },
           "model_cache_size": len(model_cache)
       }), 200
   except Exception as e:
       print(f"Error getting status: {str(e)}")
       return jsonify({"error": str(e)}), 500

# New endpoint to test if a custom prompt exists
@app.route('/quiz/check-prompt', methods=['GET'])
@async_to_sync
async def check_prompt():
   try:
       standard = request.args.get('standard', '')
       subject = request.args.get('subject', '')
       chapter = request.args.get('chapter', '')
       is_practice_mode = request.args.get('is_practice_mode', 'true').lower() == 'true'
       
       if not standard or not subject or not chapter:
           return jsonify({"error": "Missing required parameters"}), 400
           
       custom_prompt_data = await get_custom_prompt(standard, subject, chapter, is_practice_mode)
       
       return jsonify({
           "has_custom_prompt": custom_prompt_data is not None,
           "mode": "Practice" if is_practice_mode else "Test",
           "total_questions": custom_prompt_data.get('total_questions', TEST_MODE_QUESTIONS) if custom_prompt_data else TEST_MODE_QUESTIONS,
           "number_of_sets": custom_prompt_data.get('number_of_sets', TEST_MODE_SETS) if custom_prompt_data else TEST_MODE_SETS,
           "time_in_minutes": custom_prompt_data.get('time_in_minutes', 30) if custom_prompt_data else 30
       }), 200
       
   except Exception as e:
       print(f"Error checking prompt: {str(e)}")
       return jsonify({"error": str(e)}), 500

# New endpoint to get available models
@app.route('/quiz/available-models', methods=['GET'])
@async_to_sync
async def get_available_models():
   try:
       if db is None:
           return jsonify({"error": "Firestore not connected"}), 500
           
       models = []
       try:
           # Get all provider documents
           providers = db.collection('models').get()
           
           for provider in providers:
               provider_id = provider.id
               # Get all model documents for this provider
               model_docs = db.collection('models').document(provider_id).collection('models').get()
               
               for model_doc in model_docs:
                   model_data = model_doc.to_dict()
                   models.append({
                       "provider": provider_id,
                       "model_id": model_doc.id,
                       "name": model_data.get("name", model_doc.id),
                       "description": model_data.get("description", f"{provider_id}/{model_doc.id}"),
                       "is_active": model_data.get("is_active", False)
                   })
       except Exception as e:
           print(f"Error fetching models from Firestore: {str(e)}")
           
       # Get current active model
       try:
           active_model_ref = db.collection('config').document('active_model').get()
           active_model = None
           if active_model_ref.exists:
               active_model_data = active_model_ref.to_dict()
               active_model = {
                   "provider": active_model_data.get("provider"),
                   "model": active_model_data.get("model")
               }
       except Exception as e:
           print(f"Error fetching active model: {str(e)}")
           active_model = None
           
       return jsonify({
           "models": models,
           "active_model": active_model
       }), 200
       
   except Exception as e:
       print(f"Error getting available models: {str(e)}")
       return jsonify({"error": str(e)}), 500

# Endpoint to set the active model
@app.route('/quiz/set-active-model', methods=['POST'])
@async_to_sync
async def set_active_model():
   try:
       if db is None:
           return jsonify({"error": "Firestore not connected"}), 500
           
       data = request.json
       provider = data.get('provider')
       model = data.get('model')
       
       if not provider or not model:
           return jsonify({"error": "Provider and model are required"}), 400
           
       # Verify the model exists
       model_ref = db.collection('models').document(provider).collection('models').document(model).get()
       if not model_ref.exists:
           return jsonify({"error": f"Model {provider}/{model} not found"}), 404
           
       # Set as active model
       db.collection('config').document('active_model').set({
           "provider": provider,
           "model": model,
           "updated_at": firestore.SERVER_TIMESTAMP
       })
       
       # Clear model cache to force reload
       model_cache.clear()
       
       return jsonify({
           "status": "success",
           "message": f"Active model set to {provider}/{model}"
       }), 200
       
   except Exception as e:
       print(f"Error setting active model: {str(e)}")
       return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
   print("\nStarting Quiz Generator Server with LangChain integration...")
   print(f"Default Test Mode Questions: {TEST_MODE_QUESTIONS}")
   print(f"OPENAI_API_KEY is {'set' if OPENAI_API_KEY else 'NOT SET'}")
   print(f"Default Test Mode Set Size: {TEST_MODE_SET_SIZE}")
   print(f"Default Test Mode Sets: {TEST_MODE_SETS}")
   print(f"Practice Mode Questions Per Set: {PRACTICE_MODE_QUESTIONS_PER_SET}")
   print(f"Hard Question Percentage: {HARD_QUESTION_PERCENTAGE}%")
   print(f"Firestore Connected: {db is not None}")
   print("=" * 50)
   
   # Preload model config at startup
   asyncio.run(get_model_config())
   
   CORS(app, resources={r"/": {"origins": ""}})
   app.run(debug=True, port=5000, host='0.0.0.0')