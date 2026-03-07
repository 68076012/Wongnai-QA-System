"""Question answering generation module for Wongnai QA System.

This module implements the answer generation using large language models.
It takes retrieved context from relevant reviews and generates answers to
user questions about restaurants and food.

Key functions:
    - Load and initialize QA language model
    - Format prompts with retrieved context
    - Generate answers using LLM inference
    - Support for quantization and LoRA fine-tuning
"""

from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import MODEL_CONFIG, DEVICE


class WongnaiQAGenerator:
    """Generator class for creating natural language answers from retrieved reviews.
    
    Supports two modes:
    1. Template-based (baseline): Always works, no GPU needed
    2. LLM-based (finetuned): Uses Typhoon/OpenThaiGPT for better answers
    
    Attributes:
        use_llm: Whether to use LLM-based generation.
        model: The loaded LLM model (if use_llm=True).
        tokenizer: The tokenizer for the LLM (if use_llm=True).
        model_name: Name of the LLM model being used.
    """
    
    def __init__(self, use_llm: bool = False, model_name: Optional[str] = None):
        """Initialize the QA generator.
        
        Args:
            use_llm: Whether to use LLM-based generation. If True, loads the model.
            model_name: Name of the LLM model to use. Defaults to MODEL_CONFIG['qa_model'].
        
        Note:
            If CUDA is not available, falls back to template mode even if use_llm=True.
        """
        self.use_llm = use_llm
        self.model = None
        self.tokenizer = None
        self.model_name = model_name or MODEL_CONFIG['qa_model']
        
        if self.use_llm:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                print("Warning: CUDA not available. Falling back to template mode.")
                self.use_llm = False
                return
            
            try:
                print(f"Loading LLM model: {self.model_name}")
                
                # Configure 4-bit quantization for memory efficiency
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model with 4-bit quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                
                print(f"Successfully loaded LLM model on {DEVICE}")
                
            except Exception as e:
                print(f"Error loading LLM model: {e}")
                print("Falling back to template mode.")
                self.use_llm = False
                self.model = None
                self.tokenizer = None
    
    def _format_stars(self, rating: int) -> str:
        """Format star rating as star symbols.
        
        Args:
            rating: Star rating (1-5).
            
        Returns:
            String with star symbols (e.g., '****' for rating 4).
        """
        return '*' * rating
    
    def generate_answer_template(self, query: str, results: List[Dict]) -> str:
        """Generate answer using template-based approach.
        
        This is the baseline method that formats retrieved results
        into a readable answer without using an LLM.
        
        Args:
            query: The original user query.
            results: List of retrieved review dictionaries.
            
        Returns:
            Formatted answer string in Thai.
        """
        # Handle empty results
        if not results:
            return 'ไม่พบข้อมูลที่ตรงกับคำถามของคุณ'
        
        # Build header
        answer_parts = [
            f'จากการค้นหา "{query}" พบ {len(results)} ร้านที่เกี่ยวข้อง:\n'
        ]
        
        # Format each result
        for i, result in enumerate(results, 1):
            star_rating = result.get('star_rating', 0)
            score = result.get('score', 0.0)
            review_text = result.get('review_text', '')
            cuisine_type = result.get('cuisine_type', [])
            food_type = result.get('food_type', [])
            location = result.get('location', [])
            
            # Build result header with rating and relevance
            result_header = f'\nร้านที่ {i} (Rating: {star_rating}/5 {self._format_stars(star_rating)}, ความเกี่ยวข้อง: {score:.0%})'
            answer_parts.append(result_header)
            
            # Add metadata if available
            metadata_parts = []
            if cuisine_type:
                metadata_parts.append(f'ประเภทอาหาร: {", ".join(cuisine_type)}')
            if food_type:
                metadata_parts.append(f'แนวอาหาร: {", ".join(food_type)}')
            if location:
                metadata_parts.append(f'ที่ตั้ง: {", ".join(location)}')
            
            if metadata_parts:
                answer_parts.append(' | '.join(metadata_parts))
            
            # Add review excerpt (first 300 chars)
            excerpt = review_text[:300].strip()
            if len(review_text) > 300:
                excerpt += '...'
            answer_parts.append(f'รีวิว: {excerpt}')
            
            # Add separator line
            answer_parts.append('-' * 50)
        
        return '\n'.join(answer_parts)
    
    def generate_answer_llm(self, query: str, results: List[Dict]) -> str:
        """Generate answer using LLM-based approach.
        
        Uses a Thai language model to generate a natural answer
        based on the retrieved review context.
        
        Args:
            query: The original user query.
            results: List of retrieved review dictionaries.
            
        Returns:
            Generated answer string in Thai.
        """
        # Handle empty results
        if not results:
            return 'ไม่พบข้อมูลที่ตรงกับคำถามของคุณ'
        
        # Handle case where model failed to load
        if self.model is None or self.tokenizer is None:
            return self.generate_answer_template(query, results)
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(results, 1):
            star_rating = result.get('star_rating', 0)
            review_text = result.get('review_text', '')
            context_parts.append(f'รีวิว {i} (Rating: {star_rating}/5): {review_text[:400]}')
        
        context = '\n\n'.join(context_parts)
        
        # Build prompt in Thai
        system_prompt = (
            'คุณเป็นผู้เชี่ยวชาญแนะนำร้านอาหาร ตอบคำถามจากข้อมูลรีวิวที่ให้มา '
            'ตอบเป็นภาษาไทย กระชับ ให้ข้อมูลครบ ระบุ star rating ของแต่ละร้านด้วย'
        )
        
        user_prompt = f'คำถาม: {query}\n\nข้อมูลรีวิว:\n{context}'
        
        # Format as chat
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        
        try:
            # Tokenize input
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            # Fallback to template mode on error
            return self.generate_answer_template(query, results)
    
    def generate_answer(self, query: str, results: List[Dict]) -> str:
        """Generate answer based on query and retrieved results.
        
        Routes to either LLM-based or template-based generation
        depending on configuration and model availability.
        
        Args:
            query: The original user query.
            results: List of retrieved review dictionaries.
            
        Returns:
            Generated answer string.
        """
        if self.use_llm and self.model is not None:
            return self.generate_answer_llm(query, results)
        else:
            return self.generate_answer_template(query, results)
    
    def format_results_for_display(
        self,
        query: str,
        results: List[Dict],
        answer: str,
    ) -> Dict:
        """Format results for web UI display.
        
        Returns a structured dictionary containing all information
        needed for the frontend to display the answer and results.
        
        Args:
            query: The original user query.
            results: List of retrieved review dictionaries.
            answer: The generated answer string.
            
        Returns:
            Dictionary with query, answer, and formatted results.
        """
        # Format each result for display
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = {
                'rank': i,
                'review_text': result.get('review_text', ''),
                'star_rating': result.get('star_rating', 0),
                'star_display': self._format_stars(result.get('star_rating', 0)),
                'relevance_score': result.get('score', 0.0),
                'cuisine_type': result.get('cuisine_type', []),
                'food_type': result.get('food_type', []),
                'atmosphere': result.get('atmosphere', []),
                'price_level': result.get('price_level', []),
                'location': result.get('location', []),
                'mentioned_foods': result.get('mentioned_foods', []),
            }
            formatted_results.append(formatted_result)
        
        return {
            'query': query,
            'answer': answer,
            'num_results': len(results),
            'results': formatted_results,
        }
