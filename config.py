"""
Configuration file for Agentic Data Analyst
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-3.5-turbo" for faster/cheaper

# Application Settings
MAX_SAMPLE_VALUES = 5  # Number of sample values to show LLM
MAX_ROWS_FOR_ANALYSIS = 10000  # Limit for large files

# File paths
DATA_DIR = "data"
OUTPUT_DIR = "output"

# Validation
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Create a .env file with OPENAI_API_KEY=your_key")