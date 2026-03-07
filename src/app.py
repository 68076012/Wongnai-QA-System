"""Gradio web UI for Wongnai QA System.

This module provides a beautiful, modern web interface for the restaurant
question-answering system with Thai language support.
"""

import os
from typing import Dict, List, Optional, Tuple

import gradio as gr

from src.config import (
    FAISS_INDEX_PATH,
    FINETUNED_FAISS_INDEX_PATH,
    MODEL_CONFIG,
)
from src.retrieval import WongnaiRetriever
from src.qa_generator import WongnaiQAGenerator

# Global variables for loaded components
_system_components: Optional[Dict] = None

# Demo queries organized by category
DEMO_QUERIES = {
    "อาหารทะเล": [
        "อาหารทะเลสดๆ แถวพัทยา ราคาไม่แพง",
        "ร้านกุ้งเผาอร่อยในกรุงเทพ",
        "หอยนางรมสด ร้านไหนดี",
    ],
    "อาหารญี่ปุ่น": [
        "ซูชิราคาถูก คุณภาพดี",
        "ร้านราเมนเจ้าอร่อย",
        "บุฟเฟ่ต์อาหารญี่ปุ่น",
    ],
    "อาหารไทย": [
        "ต้มยำกุ้งรสเด็ด",
        "ร้านผัดไทยอร่อยๆ",
        "ข้าวซอยเชียงใหม่แท้ๆ",
    ],
    "บรรยากาศ": [
        "ร้านอาหารบรรยากาศดี วิวสวย",
        "คาเฟ่เงียบสงบ นั่งทำงาน",
        "ร้านอาหารสำหรับเดท",
    ],
    "ราคา": [
        "อาหารอร่อยราคาถูก",
        "บุฟเฟ่ต์ราคาหลักร้อย",
        "ร้านหรูราคาไม่แพง",
    ],
}

# Filter options
CUISINE_OPTIONS = ["", "ไทย", "ญี่ปุ่น", "จีน", "เกาหลี", "อิตาเลียน", "ฝรั่งเศส", "อินเดีย", "เวียดนาม"]
FOOD_OPTIONS = ["", "อาหารทะเล", "พิซซ่า", "ก๋วยเตี๋ยว", "กาแฟ", "เบเกอรี่", "สเต๊ก", "แกง", "ของหวาน"]
RATING_OPTIONS = [("ไม่กำหนด", None), ("1 ดาว", 1), ("2 ดาว", 2), ("3 ดาว", 3), ("4 ดาว", 4), ("5 ดาว", 5)]

# Custom CSS for Thai-friendly styling
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');

:root {
    --primary-orange: #ff6b35;
    --primary-red: #e63946;
    --warm-bg: #fff8f5;
    --card-bg: #ffffff;
    --text-dark: #2d3436;
    --text-light: #636e72;
    --border-color: #ffe5dc;
}

* {
    font-family: 'Sarabun', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body {
    background: linear-gradient(135deg, #fff8f5 0%, #fff0eb 100%) !important;
}

.gradio-container {
    max-width: 1400px !important;
    background: transparent !important;
}

/* Header styling */
.header-title {
    text-align: center;
    padding: 2rem 0 1rem 0;
    background: linear-gradient(135deg, var(--primary-orange) 0%, var(--primary-red) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.header-subtitle {
    text-align: center;
    color: var(--text-light);
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Card styling */
.result-card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(255, 107, 53, 0.1);
    border: 1px solid var(--border-color);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.result-rank {
    background: linear-gradient(135deg, var(--primary-orange), var(--primary-red));
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.9rem;
}

.result-score {
    background: #fff3e0;
    color: var(--primary-orange);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}

/* Star rating styling */
.star-rating {
    color: #ffc107;
    font-size: 1.1rem;
    letter-spacing: 2px;
}

.star-empty {
    color: #e0e0e0;
}

/* Answer box styling */
.answer-box {
    background: linear-gradient(135deg, #fff8f5 0%, #ffffff 100%);
    border-left: 4px solid var(--primary-orange);
    padding: 1.5rem;
    border-radius: 0 12px 12px 0;
    margin: 1rem 0;
}

/* Comparison styling */
.compare-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
}

.compare-box {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.5rem;
    border: 2px solid var(--border-color);
}

.compare-box.baseline {
    border-color: #74b9ff;
}

.compare-box.finetuned {
    border-color: #00b894;
}

/* Demo query buttons */
.demo-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.demo-category {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid var(--border-color);
}

.demo-category h4 {
    color: var(--primary-orange);
    margin-bottom: 0.75rem;
    font-weight: 600;
}

/* Button styling */
button.primary {
    background: linear-gradient(135deg, var(--primary-orange) 0%, var(--primary-red) 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Input styling */
input[type="text"], textarea {
    border-radius: 8px !important;
    border: 2px solid var(--border-color) !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: var(--primary-orange) !important;
    box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.1) !important;
}

/* Accordion styling */
.accordion {
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* Metadata tags */
.metadata-tag {
    display: inline-block;
    background: #f0f0f0;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.8rem;
    margin: 0.2rem;
    color: var(--text-dark);
}

.metadata-tag.cuisine {
    background: #e3f2fd;
    color: #1976d2;
}

.metadata-tag.food {
    background: #f3e5f5;
    color: #7b1fa2;
}

.metadata-tag.location {
    background: #e8f5e9;
    color: #388e3c;
}

/* About page styling */
.about-section {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid var(--border-color);
}

.about-section h3 {
    color: var(--primary-orange);
    margin-bottom: 1rem;
}

.architecture-diagram {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1.5rem;
    font-family: monospace !important;
    text-align: center;
    margin: 1rem 0;
}

/* Stats grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.stat-card {
    background: linear-gradient(135deg, #fff8f5 0%, #ffffff 100%);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    border: 1px solid var(--border-color);
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary-orange);
}

.stat-label {
    font-size: 0.85rem;
    color: var(--text-light);
}

/* Error message */
.error-message {
    background: #ffebee;
    color: #c62828;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #c62828;
}

/* Loading indicator */
.loading-text {
    color: var(--primary-orange);
    font-style: italic;
}
"""


def format_stars_html(rating: int) -> str:
    """Format star rating as HTML with star symbols."""
    stars = "★" * rating
    empty = "☆" * (5 - rating)
    return f'<span class="star-rating">{stars}</span><span class="star-empty">{empty}</span>'


def format_review_card(result: Dict, rank: int) -> str:
    """Format a single review result as an HTML card."""
    star_html = format_stars_html(result.get("star_rating", 0))
    score = result.get("relevance_score", 0)
    
    # Build metadata tags
    tags_html = ""
    
    cuisines = result.get("cuisine_type", [])
    if cuisines:
        for c in cuisines[:3]:
            tags_html += f'<span class="metadata-tag cuisine">{c}</span>'
    
    foods = result.get("food_type", [])
    if foods:
        for f in foods[:3]:
            tags_html += f'<span class="metadata-tag food">{f}</span>'
    
    locations = result.get("location", [])
    if locations:
        for l in locations[:2]:
            tags_html += f'<span class="metadata-tag location">{l}</span>'
    
    # Truncate review text
    review_text = result.get("review_text", "")
    if len(review_text) > 400:
        review_text = review_text[:400] + "..."
    
    return f"""
    <div class="result-card">
        <div class="result-header">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div class="result-rank">{rank}</div>
                <div>{star_html} <span style="color: #666; font-size: 0.9rem;">({result.get("star_rating", 0)}/5)</span></div>
            </div>
            <div class="result-score">ความเกี่ยวข้อง: {score:.1%}</div>
        </div>
        <div style="margin-bottom: 0.75rem;">{tags_html}</div>
        <div style="color: #444; line-height: 1.6;">{review_text}</div>
    </div>
    """


def format_answer_html(answer: str, query: str) -> str:
    """Format the generated answer as HTML."""
    return f"""
    <div class="answer-box">
        <div style="font-weight: 600; color: var(--primary-orange); margin-bottom: 0.5rem;">คำตอบสำหรับ: "{query}"</div>
        <div style="line-height: 1.8; color: var(--text-dark);">{answer.replace(chr(10), "<br>")}</div>
    </div>
    """


def load_system() -> Dict:
    """Load all system components (retrievers and QA generators).
    
    Returns:
        Dictionary containing loaded retrievers and generators.
    """
    global _system_components
    
    if _system_components is not None:
        return _system_components
    
    components = {
        "baseline_retriever": None,
        "finetuned_retriever": None,
        "qa_generator": None,
        "llm_generator": None,
        "loaded": False,
        "error_message": None,
    }
    
    try:
        # Load baseline retriever
        if os.path.exists(FAISS_INDEX_PATH):
            components["baseline_retriever"] = WongnaiRetriever()
            components["baseline_retriever"].load_index()
        else:
            components["error_message"] = f"Baseline index not found at {FAISS_INDEX_PATH}. Please build the index first."
            return components
        
        # Load finetuned retriever if available
        if os.path.exists(FINETUNED_FAISS_INDEX_PATH):
            components["finetuned_retriever"] = WongnaiRetriever(
                model_name=MODEL_CONFIG["embedding_model"],
                index_path=FINETUNED_FAISS_INDEX_PATH,
                is_finetuned=True,
            )
            components["finetuned_retriever"].load_index()
        
        # Initialize QA generators
        components["qa_generator"] = WongnaiQAGenerator(use_llm=False)
        
        # Try to load LLM generator (may fail if no GPU)
        try:
            components["llm_generator"] = WongnaiQAGenerator(use_llm=True)
        except Exception as e:
            print(f"LLM generator not available: {e}")
            components["llm_generator"] = None
        
        components["loaded"] = True
        
    except Exception as e:
        components["error_message"] = f"Error loading system: {str(e)}"
    
    _system_components = components
    return components


def search_and_answer(
    query: str,
    num_results: int,
    mode: str,
    min_rating: Optional[int],
    cuisine: str,
    food_type: str,
) -> Tuple[str, str]:
    """Search and generate answer based on user query.
    
    Args:
        query: User's search query.
        num_results: Number of results to retrieve.
        mode: Search mode ('baseline', 'finetuned', or 'compare').
        min_rating: Minimum star rating filter.
        cuisine: Cuisine type filter.
        food_type: Food type filter.
        
    Returns:
        Tuple of (answer_html, results_html) for display.
    """
    components = load_system()
    
    if not components["loaded"]:
        error_html = f'<div class="error-message">{components.get("error_message", "System not loaded")}</div>'
        return error_html, ""
    
    if not query.strip():
        return "<div class='error-message'>กรุณาป้อนคำถามหรือคำค้นหา</div>", ""
    
    # Prepare filters
    cuisine_filter = cuisine if cuisine else None
    food_filter = food_type if food_type else None
    rating_filter = min_rating if min_rating else None
    
    try:
        if mode == "compare":
            return _search_compare_mode(
                query, num_results, components,
                rating_filter, cuisine_filter, food_filter
            )
        else:
            return _search_single_mode(
                query, num_results, mode, components,
                rating_filter, cuisine_filter, food_filter
            )
    except Exception as e:
        error_html = f'<div class="error-message">เกิดข้อผิดพลาด: {str(e)}</div>'
        return error_html, ""


def _search_single_mode(
    query: str,
    num_results: int,
    mode: str,
    components: Dict,
    min_rating: Optional[int],
    cuisine: Optional[str],
    food_type: Optional[str],
) -> Tuple[str, str]:
    """Search using a single model mode."""
    # Select retriever
    if mode == "finetuned" and components["finetuned_retriever"]:
        retriever = components["finetuned_retriever"]
    else:
        retriever = components["baseline_retriever"]
    
    # Search with filters
    results = retriever.search_with_filters(
        query=query,
        top_k=num_results,
        min_rating=min_rating,
        cuisine_filter=cuisine,
        food_type_filter=food_type,
    )
    
    # Generate answer
    if components["llm_generator"] and mode == "finetuned":
        answer = components["llm_generator"].generate_answer(query, results)
    else:
        answer = components["qa_generator"].generate_answer(query, results)
    
    # Format output
    answer_html = format_answer_html(answer, query)
    
    results_html = ""
    if not results:
        results_html = '<div class="result-card"><p>ไม่พบผลลัพธ์ที่ตรงกับเงื่อนไข</p></div>'
    else:
        for i, result in enumerate(results, 1):
            results_html += format_review_card(result, i)
    
    return answer_html, results_html


def _search_compare_mode(
    query: str,
    num_results: int,
    components: Dict,
    min_rating: Optional[int],
    cuisine: Optional[str],
    food_type: Optional[str],
) -> Tuple[str, str]:
    """Search using both models for comparison."""
    # Get baseline results
    baseline_results = components["baseline_retriever"].search_with_filters(
        query=query,
        top_k=num_results,
        min_rating=min_rating,
        cuisine_filter=cuisine,
        food_type_filter=food_type,
    )
    baseline_answer = components["qa_generator"].generate_answer(query, baseline_results)
    
    # Get finetuned results if available
    if components["finetuned_retriever"]:
        finetuned_results = components["finetuned_retriever"].search_with_filters(
            query=query,
            top_k=num_results,
            min_rating=min_rating,
            cuisine_filter=cuisine,
            food_type_filter=food_type,
        )
        if components["llm_generator"]:
            finetuned_answer = components["llm_generator"].generate_answer(query, finetuned_results)
        else:
            finetuned_answer = components["qa_generator"].generate_answer(query, finetuned_results)
    else:
        finetuned_results = baseline_results
        finetuned_answer = "(โมเดล Fine-tuned ไม่พร้อมใช้งาน)"
    
    # Format comparison output
    answer_html = f"""
    <div class="compare-container">
        <div class="compare-box baseline">
            <h4 style="color: #1976d2; margin-bottom: 1rem;">Baseline Model</h4>
            <div style="line-height: 1.8;">{baseline_answer.replace(chr(10), "<br>")}</div>
        </div>
        <div class="compare-box finetuned">
            <h4 style="color: #388e3c; margin-bottom: 1rem;">Fine-tuned Model</h4>
            <div style="line-height: 1.8;">{finetuned_answer.replace(chr(10), "<br>")}</div>
        </div>
    </div>
    """
    
    # Format results comparison
    results_html = "<h4 style='margin: 1.5rem 0 1rem 0; color: var(--primary-orange);'>ผลลัพธ์จาก Baseline Model:</h4>"
    if baseline_results:
        for i, result in enumerate(baseline_results, 1):
            results_html += format_review_card(result, i)
    else:
        results_html += '<div class="result-card"><p>ไม่พบผลลัพธ์</p></div>'
    
    if components["finetuned_retriever"]:
        results_html += "<h4 style='margin: 1.5rem 0 1rem 0; color: var(--primary-orange);'>ผลลัพธ์จาก Fine-tuned Model:</h4>"
        if finetuned_results:
            for i, result in enumerate(finetuned_results, 1):
                results_html += format_review_card(result, i)
        else:
            results_html += '<div class="result-card"><p>ไม่พบผลลัพธ์</p></div>'
    
    return answer_html, results_html


def run_demo_query(query: str) -> Tuple[str, str, str, int, str]:
    """Run a demo query and return values for UI update."""
    return query, 5, "compare", "", ""


def create_about_tab() -> gr.Tab:
    """Create the About tab with system information."""
    with gr.Tab("เกี่ยวกับระบบ"):
        gr.Markdown("""
        <div class="about-section">
            <h3>Wongnai QA System - ระบบถาม-ตอบร้านอาหารอัจฉริยะ</h3>
            <p style="line-height: 1.8;">
                ระบบถาม-ตอบร้านอาหารที่ใช้เทคโนโลยี Natural Language Processing (NLP) 
                และ Large Language Models (LLM) เพื่อช่วยค้นหาและแนะนำร้านอาหารจากรีวิวจริงบน Wongnai
            </p>
        </div>
        """)
        
        gr.Markdown("""
        <div class="about-section">
            <h3>สถาปัตยกรรมระบบ</h3>
            <div class="architecture-diagram">
                <pre style="margin: 0; font-size: 0.9rem;">
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│  Query Encoder   │────▶│   FAISS Index   │
│  (คำถามผู้ใช้)   │     │ (Sentence Transformer)│  (Vector DB)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                           ┌───────────────────────────────┘
                           ▼
                    ┌──────────────────┐
                    │ Similarity Search │
                    │  (ค้นหาเชิงความหมาย)│
                    └──────────────────┘
                           │
                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Generated      │◀────│  Answer Generator │◀────│ Retrieved       │
│  Answer         │     │  (LLM/Template)   │     │ Reviews         │
│  (คำตอบ)        │     │                   │     │ (รีวิวที่ค้นพบ)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                </pre>
            </div>
        </div>
        """)
        
        gr.Markdown("""
        <div class="about-section">
            <h3>ข้อมูลโมเดล</h3>
            <ul style="line-height: 2;">
                <li><strong>Embedding Model:</strong> intfloat/multilingual-e5-base (หรือโมเดลที่ Fine-tune แล้ว)</li>
                <li><strong>QA Model:</strong> scb10x/llama3.1-typhoon2-8b-instruct (Thai LLM)</li>
                <li><strong>Vector Database:</strong> FAISS (Facebook AI Similarity Search)</li>
                <li><strong>Framework:</strong> Hugging Face Transformers, Sentence-Transformers</li>
            </ul>
        </div>
        """)
        
        gr.Markdown("""
        <div class="about-section">
            <h3>สถิติชุดข้อมูล</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">100K+</div>
                    <div class="stat-label">รีวิวร้านอาหาร</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">10K+</div>
                    <div class="stat-label">ร้านอาหาร</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">20+</div>
                    <div class="stat-label">ประเภทอาหาร</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">77</div>
                    <div class="stat-label">จังหวัดทั่วไทย</div>
                </div>
            </div>
        </div>
        """)


def create_demo_tab() -> gr.Tab:
    """Create the Demo Queries tab."""
    with gr.Tab("ตัวอย่างคำถ"):
        gr.Markdown("### เลือกคำถามตัวอย่างเพื่อทดสอบระบบ")
        gr.Markdown("คลิกที่คำถามใดก็ได้เพื่อรันคำค้นหาและเปรียบเทียบผลลัพธ์")
        
        for category, queries in DEMO_QUERIES.items():
            with gr.Row():
                gr.Markdown(f"**{category}**")
            with gr.Row():
                for query in queries:
                    btn = gr.Button(query, size="sm")
                    btn.click(
                        fn=run_demo_query,
                        inputs=[],
                        outputs=[query_input, num_results, model_mode, cuisine_filter, food_filter]
                    )


def launch_app(share: bool = False) -> None:
    """Build and launch the Gradio web interface.
    
    Args:
        share: Whether to create a public shareable link.
    """
    # Load system components on startup
    components = load_system()
    
    # Custom theme with warm food-related colors
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="red",
        neutral_hue="stone",
    ).set(
        body_background_fill="linear-gradient(135deg, #fff8f5 0%, #fff0eb 100%)",
        button_primary_background_fill="linear-gradient(135deg, #ff6b35 0%, #e63946 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #e55a2b 0%, #d32f3f 100%)",
        button_primary_text_color="white",
        slider_color="#ff6b35",
    )
    
    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="Wongnai QA System",
    ) as demo:
        # Header
        gr.Markdown("""
        <h1 class="header-title">Wongnai QA System</h1>
        <p class="header-subtitle">ระบบถาม-ตอบร้านอาหารอัจฉริยะ ใช้ AI ช่วยค้นหาและแนะนำร้านอาหารจากรีวิวจริง</p>
        """)
        
        # Check system status
        if not components["loaded"]:
            gr.Markdown(f"""
            <div class="error-message">
                ⚠️ {components.get("error_message", "ระบบยังไม่พร้อมใช้งาน")}<br>
                กรุณาสร้าง FAISS index ก่อนใช้งาน (รันคำสั่ง: python src/retrieval.py)
            </div>
            """)
        
        with gr.Tab("ค้นหา"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Query input
                    query_input = gr.Textbox(
                        label="ถามเกี่ยวกับร้านอาหาร",
                        placeholder="เช่น อาหารทะเลสดๆ แถวพัทยา ราคาไม่แพง",
                        lines=2,
                    )
                    
                    with gr.Row():
                        num_results = gr.Slider(
                            label="จำนวนผลลัพธ์",
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                        )
                        model_mode = gr.Radio(
                            label="โหมดการค้นหา",
                            choices=[
                                ("Baseline Model", "baseline"),
                                ("Fine-tuned Model", "finetuned"),
                                ("เปรียบเทียบทั้งสอง", "compare"),
                            ],
                            value="baseline",
                        )
                    
                    # Optional filters
                    with gr.Accordion("ตัวกรองเพิ่มเติม (คลิกเพื่อขยาย)", open=False):
                        with gr.Row():
                            min_rating = gr.Dropdown(
                                label="คะแนนขั้นต่ำ",
                                choices=RATING_OPTIONS,
                                value=None,
                            )
                            cuisine_filter = gr.Dropdown(
                                label="ประเภทอาหาร",
                                choices=CUISINE_OPTIONS,
                                value="",
                            )
                            food_filter = gr.Dropdown(
                                label="แนวอาหาร",
                                choices=FOOD_OPTIONS,
                                value="",
                            )
                    
                    # Search button
                    search_btn = gr.Button("🔍 ค้นหา", variant="primary", size="lg")
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["อาหารทะเลสดๆ แถวพัทยา", 5, "baseline", None, "", ""],
                            ["ร้านราเมนอร่อยในกรุงเทพ", 5, "finetuned", None, "ญี่ปุ่น", "ก๋วยเตี๋ยว"],
                            ["คาเฟ่บรรยากาศดี น่านั่งทำงาน", 5, "compare", 4, "", "กาแฟ"],
                            ["บุฟเฟ่ต์อาหารญี่ปุ่นราคาคุ้มค่า", 5, "compare", None, "ญี่ปุ่น", ""],
                        ],
                        inputs=[query_input, num_results, model_mode, min_rating, cuisine_filter, food_filter],
                        label="ตัวอย่างคำถาม",
                    )
                
                with gr.Column(scale=3):
                    # Output areas
                    answer_output = gr.HTML(label="คำตอบ")
                    
                    with gr.Accordion("ดูรีวิวที่เกี่ยวข้อง", open=True):
                        results_output = gr.HTML()
        
        # Demo queries tab
        with gr.Tab("ตัวอย่างคำถาม"):
            gr.Markdown("### เลือกคำถามตัวอย่างเพื่อทดสอบระบบ")
            gr.Markdown("คลิกที่คำถามใดก็ได้เพื่อรันคำค้นหาและเปรียบเทียบผลลัพธ์")
            
            demo_outputs = [query_input, num_results, model_mode, cuisine_filter, food_filter]
            
            for category, queries in DEMO_QUERIES.items():
                with gr.Row():
                    gr.Markdown(f"**{category}**")
                with gr.Row():
                    for query in queries:
                        btn = gr.Button(query, size="sm", variant="secondary")
                        btn.click(
                            fn=lambda q=query: (q, 5, "compare", "", ""),
                            inputs=[],
                            outputs=demo_outputs,
                        )
        
        # About tab
        with gr.Tab("เกี่ยวกับระบบ"):
            gr.Markdown("""
            <div class="about-section">
                <h3>Wongnai QA System - ระบบถาม-ตอบร้านอาหารอัจฉริยะ</h3>
                <p style="line-height: 1.8;">
                    ระบบถาม-ตอบร้านอาหารที่ใช้เทคโนโลยี Natural Language Processing (NLP) 
                    และ Large Language Models (LLM) เพื่อช่วยค้นหาและแนะนำร้านอาหารจากรีวิวจริงบน Wongnai
                </p>
            </div>
            """)
            
            gr.Markdown("""
            <div class="about-section">
                <h3>สถาปัตยกรรมระบบ</h3>
                <div class="architecture-diagram">
                    <pre style="margin: 0; font-size: 0.85rem; overflow-x: auto;">
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   User Query    │────▶│   Query Encoder      │────▶│   FAISS Index   │
│  (คำถามผู้ใช้)   │     │ (Sentence Transformer)│    │  (Vector DB)    │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                              │
                              ┌───────────────────────────────┘
                              ▼
                       ┌──────────────────┐
                       │ Similarity Search │
                       │(ค้นหาเชิงความหมาย)│
                       └──────────────────┘
                              │
                              ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Generated      │◀────│  Answer Generator │◀────│ Retrieved       │
│  Answer         │     │  (LLM/Template)   │     │ Reviews         │
│  (คำตอบ)        │     │                   │     │ (รีวิวที่ค้นพบ)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                    </pre>
                </div>
            </div>
            """)
            
            gr.Markdown("""
            <div class="about-section">
                <h3>ข้อมูลโมเดล</h3>
                <ul style="line-height: 2;">
                    <li><strong>Embedding Model:</strong> intfloat/multilingual-e5-base</li>
                    <li><strong>QA Model:</strong> scb10x/llama3.1-typhoon2-8b-instruct (Thai LLM)</li>
                    <li><strong>Vector Database:</strong> FAISS (Facebook AI Similarity Search)</li>
                    <li><strong>Framework:</strong> Hugging Face Transformers, Sentence-Transformers</li>
                </ul>
            </div>
            """)
            
            gr.Markdown("""
            <div class="about-section">
                <h3>สถิติชุดข้อมูล</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">100K+</div>
                        <div class="stat-label">รีวิวร้านอาหาร</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">10K+</div>
                        <div class="stat-label">ร้านอาหาร</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">20+</div>
                        <div class="stat-label">ประเภทอาหาร</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">77</div>
                        <div class="stat-label">จังหวัดทั่วไทย</div>
                    </div>
                </div>
            </div>
            """)
        
        # Event handlers
        search_btn.click(
            fn=search_and_answer,
            inputs=[query_input, num_results, model_mode, min_rating, cuisine_filter, food_filter],
            outputs=[answer_output, results_output],
        )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        show_error=True,
    )


def main():
    """Main entry point to launch the app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wongnai QA System Web UI")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    args = parser.parse_args()
    
    launch_app(share=args.share)


if __name__ == "__main__":
    main()
