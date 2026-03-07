"""Data preprocessing module for Wongnai QA System.

This module handles loading, cleaning, and preprocessing of Wongnai review data,
food dictionary, and query datasets. It prepares the data for indexing and
retrieval operations.

Key functions:
    - Load and clean review CSV data
    - Process food dictionary for entity recognition
    - Prepare query datasets for training and evaluation
    - Text normalization and tokenization for Thai language
"""

import os
import re
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Set

from src.config import (
    REVIEW_TRAIN_FILE,
    FOOD_DICT_FILE,
    QUERY_JUDGES_FILE,
    QUERY_ALGO_FILE,
    PROCESSED_DATA_PATH,
)

# Keyword dictionaries for metadata extraction
CUISINE_KEYWORDS = {
    'thai': ['อาหารไทย', 'thai', 'ไทย'],
    'chinese': ['อาหารจีน', 'chinese', 'จีน'],
    'japanese': ['อาหารญี่ปุ่น', 'japanese', 'ญี่ปุ่น', 'เทปันยากิ'],
    'korean': ['อาหารเกาหลี', 'korean', 'เกาหลี'],
    'indian': ['อาหารอินเดีย', 'indian', 'อินเดีย'],
    'italian': ['อาหารอิตาลี', 'italian', 'อิตาลี'],
    'french': ['อาหารฝรั่งเศส', 'french', 'ฝรั่งเศส'],
    'vietnamese': ['อาหารเวียดนาม', 'vietnamese', 'เวียดนาม'],
    'mexican': ['mexican', 'เม็กซิกัน', 'เม็กซิโก'],
    'western': ['western', 'ฝรั่ง', 'อาหารฝรั่ง'],
    'fusion': ['ฟิวชั่น', 'fusion'],
    'isan': ['อีสาน', 'อิสาน', 'isan'],
    'southern': ['ใต้', 'southern thai', 'ภาคใต้'],
    'northern': ['เหนือ', 'northern thai', 'ภาคเหนือ'],
}

FOOD_TYPE_KEYWORDS = {
    'seafood': ['อาหารทะเล', 'ซีฟู้ด', 'seafood', 'ทะเล'],
    'pizza': ['พิซซ่า', 'pizza', 'พิซซา'],
    'bakery': ['เบเกอรี่', 'bakery', 'bakeries'],
    'dessert': ['ขนม', 'dessert', 'ของหวาน', 'ขนมหวาน'],
    'beverage': ['เครื่องดื่ม', 'beverage', 'drink'],
    'ice_cream': ['ไอศกรีม', 'ice cream', 'icecream'],
    'rice_curry': ['ข้าวแกง', 'แกงถุง', 'ข้าวราดแกง'],
    'noodle': ['ก๋วยเตี๋ยว', 'noodle', 'ก๋วยเตี๋ยว', 'เส้น', 'ขนมจีน'],
    'a_la_carte': ['ตามสั่ง', 'อาหารตามสั่ง'],
    'healthy': ['สุขภาพ', 'healthy', 'เพื่อสุขภาพ', 'คลีน', 'clean food'],
    'buffet': ['บุฟเฟ่ต์', 'buffet', 'ปิ้งย่าง', 'หมูกระทะ'],
    'shabu': ['ชาบู', 'shabu', 'ชาบูชาบู'],
    'steak': ['สเต็ก', 'steak'],
    'coffee': ['กาแฟ', 'coffee'],
    'tea': ['ชา', 'tea'],
    'bbq': ['ปิ้งย่าง', 'BBQ', 'barbecue', 'บาร์บีคิว'],
    'sushi': ['ซูชิ', 'sushi'],
    'ramen': ['ราเมน', 'ramen'],
    'dim_sum': ['ติ่มซำ', 'dim sum', 'dimsum'],
    'grill': ['หมูกระทะ', 'ย่าง', 'grill'],
    'salad': ['สลัด', 'salad'],
    'soup': ['ซุป', 'soup', 'น้ำซุป'],
}

ATMOSPHERE_KEYWORDS = {
    'luxury': ['หรูหรา', 'luxury', 'luxurious', 'หรู', 'hi-so'],
    'air_con': ['ติดแอร์', 'แอร์', 'air-con', 'aircon', 'air con'],
    'open_air': ['ร้านเปิด', 'open air', 'open-air', 'outdoor'],
    'street_food': ['ร้านข้างทาง', 'street food', 'ข้างทาง', 'แผงลอย'],
    'good_atmosphere': ['บรรยากาศดี', 'atmosphere', 'บรรยากาศ'],
    'good_view': ['วิวสวย', 'view', 'ทิวทัศน์'],
    'waterfront': ['ริมน้ำ', 'ริมคลอง', 'ริมแม่น้ำ', 'waterfront'],
    'beachfront': ['ริมทะเล', 'beachfront', 'beach', 'ทะเล'],
    'quiet': ['สงบ', 'quiet', 'เงียบ', 'ส่วนตัว', 'private'],
    'romantic': ['โรแมนติก', 'romantic', 'เดท', 'date'],
    'cafe': ['คาเฟ่', 'cafe', 'café'],
    'instagrammable': ['instagrammable', 'ig', 'ไอจี', 'ถ่ายรูปสวย', 'รูปสวย'],
    'cozy': ['น่านั่ง', 'cozy', 'comfortable', 'สบาย'],
    'spacious': ['กว้างขวาง', 'spacious', 'กว้าง', 'ใหญ่'],
    'family': ['ครอบครัว', 'family', 'เด็ก', 'kids', 'กลุ่มใหญ่'],
}

PRICE_KEYWORDS = {
    'expensive': ['ราคาแพง', 'แพง', 'expensive', 'pricey', 'high price'],
    'cheap': ['ราคาย่อมเยา', 'ถูก', 'ราคาถูก', 'cheap', 'inexpensive', 'ราคานักเรียน'],
    'worth': ['คุ้มค่า', 'คุ้ม', 'worth', 'value for money'],
    'reasonable': ['ราคาเหมาะสม', 'reasonable', 'fair price', 'เหมาะสม'],
    'premium': ['premium', 'พรีเมี่ยม', 'พรีเมียม'],
    'budget': ['ไม่แพง', 'budget', 'ราคาประหยัด', 'ประหยัด'],
}

LOCATION_KEYWORDS = {
    'beachside': ['ติดทะเล', 'ริมทะเล', 'beachside'],
    'mountain': ['บนเขา', 'mountain', 'ภูเขา', 'เขา'],
    'downtown': ['ใจกลางเมือง', 'downtown', 'city center', 'cbd'],
    'suburb': ['ชานเมือง', 'suburb', 'รอบนอก'],
    'mall': ['ห้างสรรพสินค้า', 'mall', 'shopping mall', 'ห้าง', 'department store'],
    'market': ['ตลาดนัด', 'market', 'ตลาด'],
}

PROVINCE_NAMES = {
    'bangkok': ['กรุงเทพ', 'bangkok', 'กรุงเทพมหานคร', 'bkk'],
    'chiang_mai': ['เชียงใหม่', 'chiang mai', 'chiangmai'],
    'pattaya': ['พัทยา', 'pattaya'],
    'phuket': ['ภูเก็ต', 'phuket'],
    'hua_hin': ['หัวหิน', 'hua hin', 'huahin'],
    'khao_yai': ['เขาใหญ่', 'khao yai', 'khaoyai'],
    'ayutthaya': ['อยุธยา', 'ayutthaya'],
    'kanchanaburi': ['กาญจนบุรี', 'kanchanaburi'],
    'rayong': ['ระยอง', 'rayong'],
    'chonburi': ['ชลบุรี', 'chonburi'],
    'korat': ['นครราชสีมา', 'โคราช', 'korat', 'nakhon ratchasima'],
    'khon_kaen': ['ขอนแก่น', 'khon kaen', 'khonkaen'],
    'songkhla': ['สงขลา', 'songkhla', 'หาดใหญ่', 'hat yai', 'hatyai'],
    'chiang_rai': ['เชียงราย', 'chiang rai', 'chiangrai'],
    'surat': ['สุราษฎร์ธานี', 'surat thani'],
    'samui': ['สมุย', 'samui', 'koh samui', 'เกาะสมุย'],
    'krabi': ['กระบี่', 'krabi'],
    'trat': ['ตราด', 'trat'],
    'chantaburi': ['จันทบุรี', 'chantaburi', 'chanthaburi'],
    'nakhon_pathom': ['นครปฐม', 'nakhon pathom'],
    'ratchaburi': ['ราชบุรี', 'ratchaburi'],
    'lampang': ['ลำปาง', 'lampang'],
    'nan': ['น่าน', 'nan'],
    'mae_hong_son': ['แม่ฮ่องสอน', 'mae hong son', 'maehongson'],
    'sukhothai': ['สุโขทัย', 'sukhothai'],
    'phitsanulok': ['พิษณุโลก', 'phitsanulok'],
    'udon_thani': ['อุดรธานี', 'udon thani', 'udonthani'],
    'sakon_nakhon': ['สกลนคร', 'sakon nakhon', 'sakonnakhon'],
    'nakhon_nayok': ['นครนายก', 'nakhon nayok'],
    'prachuap': ['ประจวบคีรีขันธ์', 'prachuap khiri khan', 'prachuapkhirikhan', 'หัวหิน'],
}


def load_reviews(file_path: str) -> pd.DataFrame:
    """Load review data from CSV file.
    
    Args:
        file_path: Path to the review CSV file.
        
    Returns:
        DataFrame with columns ['review_text', 'star_rating'].
    """
    df = pd.read_csv(file_path, sep=';', header=None, names=['review_text', 'star_rating'])
    df = df.dropna(subset=['review_text'])
    df['star_rating'] = df['star_rating'].astype(int)
    return df


def load_food_dictionary(file_path: str) -> List[str]:
    """Load food dictionary from text file.
    
    Args:
        file_path: Path to the food dictionary file.
        
    Returns:
        List of food names.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        foods = [line.strip() for line in f if line.strip()]
    return foods


def load_queries(file_path: str) -> List[str]:
    """Load query data from text file.
    
    Args:
        file_path: Path to the query file.
        
    Returns:
        List of query strings with pipe characters removed.
    """
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Remove pipe characters and clean whitespace
                query = line.replace('|', ' ')
                query = re.sub(r'\s+', ' ', query).strip()
                queries.append(query)
    return queries


def clean_review(text: str) -> str:
    """Clean and normalize review text.
    
    Args:
        text: Raw review text.
        
    Returns:
        Cleaned text, or empty string if text is too short.
    """
    if pd.isna(text):
        return ""
    
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', str(text))
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Remove very short reviews
    if len(text) < 20:
        return ""
    
    return text


def _detect_keywords(text: str, keyword_dict: Dict[str, List[str]]) -> List[str]:
    """Helper function to detect keywords from text.
    
    Args:
        text: Text to search in.
        keyword_dict: Dictionary mapping category to list of keywords.
        
    Returns:
        List of matched categories.
    """
    text_lower = text.lower()
    matched = []
    for category, keywords in keyword_dict.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(category)
    return matched


def extract_metadata(review_text: str, food_dict: Set[str]) -> Dict:
    """Extract structured metadata from review text.
    
    Args:
        review_text: The review text to analyze.
        food_dict: Set of food names for entity recognition.
        
    Returns:
        Dictionary containing extracted metadata:
            - cuisine_type: List of detected cuisine types
            - food_type: List of detected food types
            - atmosphere: List of detected atmosphere descriptors
            - price_level: List of detected price descriptors
            - location: List of detected locations
            - mentioned_foods: List of food items found in review (max 10)
    """
    if pd.isna(review_text):
        review_text = ""
    
    text = str(review_text)
    text_lower = text.lower()
    
    metadata = {
        'cuisine_type': _detect_keywords(text, CUISINE_KEYWORDS),
        'food_type': _detect_keywords(text, FOOD_TYPE_KEYWORDS),
        'atmosphere': _detect_keywords(text, ATMOSPHERE_KEYWORDS),
        'price_level': _detect_keywords(text, PRICE_KEYWORDS),
        'location': _detect_keywords(text, LOCATION_KEYWORDS),
    }
    
    # Also detect provinces as locations
    detected_provinces = _detect_keywords(text, PROVINCE_NAMES)
    metadata['location'].extend(detected_provinces)
    # Remove duplicates while preserving order
    metadata['location'] = list(dict.fromkeys(metadata['location']))
    
    # Find mentioned foods from food_dict (only check items 4+ chars, limit dict size)
    mentioned_foods = []
    for food in food_dict:
        if food.lower() in text_lower:
            mentioned_foods.append(food)
            if len(mentioned_foods) >= 5:
                break
    metadata['mentioned_foods'] = mentioned_foods
    
    return metadata


def create_search_text(row: pd.Series) -> str:
    """Create a rich search text combining review and metadata.
    
    Args:
        row: DataFrame row containing review_text and metadata columns.
        
    Returns:
        Formatted search text string.
    """
    # Get metadata values, handling both list and string formats
    cuisine = ', '.join(row.get('cuisine_type', [])) if isinstance(row.get('cuisine_type'), list) else ''
    food = ', '.join(row.get('food_type', [])) if isinstance(row.get('food_type'), list) else ''
    atmos = ', '.join(row.get('atmosphere', [])) if isinstance(row.get('atmosphere'), list) else ''
    price = ', '.join(row.get('price_level', [])) if isinstance(row.get('price_level'), list) else ''
    loc = ', '.join(row.get('location', [])) if isinstance(row.get('location'), list) else ''
    
    # Get review text (first 500 chars)
    review = str(row.get('review_text', ''))[:500]
    
    # Build search text
    parts = [p for p in [cuisine, food, atmos, price, loc] if p]
    metadata_str = ' '.join(parts)
    
    if metadata_str:
        return f"{metadata_str}: {review}"
    return review


def process_all_reviews(
    reviews_df: pd.DataFrame,
    food_dict_set: Set[str],
    max_reviews: int = 50000
) -> pd.DataFrame:
    """Process all reviews: clean, sample, extract metadata, and save.
    
    Args:
        reviews_df: DataFrame with review data.
        food_dict_set: Set of food names for entity recognition.
        max_reviews: Maximum number of reviews to process (for efficiency).
        
    Returns:
        Processed DataFrame with metadata columns.
    """
    print(f"Starting with {len(reviews_df)} reviews...")
    
    # Clean reviews
    print("Cleaning reviews...")
    tqdm.pandas(desc="Cleaning")
    reviews_df['review_text'] = reviews_df['review_text'].progress_apply(clean_review)
    
    # Filter out empty reviews
    reviews_df = reviews_df[reviews_df['review_text'] != '']
    print(f"After cleaning: {len(reviews_df)} reviews")
    
    # Sample for efficiency
    if len(reviews_df) > max_reviews:
        print(f"Sampling {max_reviews} reviews for processing...")
        reviews_df = reviews_df.sample(n=max_reviews, random_state=42)
    
    # Extract metadata
    print("Extracting metadata...")
    tqdm.pandas(desc="Metadata extraction")
    metadata_series = reviews_df['review_text'].progress_apply(
        lambda x: extract_metadata(x, food_dict_set)
    )
    
    # Convert metadata dicts to separate columns
    metadata_df = pd.DataFrame(metadata_series.tolist())
    reviews_df = pd.concat([reviews_df.reset_index(drop=True), metadata_df], axis=1)
    
    # Create search text column
    print("Creating search text...")
    reviews_df['search_text'] = reviews_df.apply(create_search_text, axis=1)
    
    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    # Save to pickle
    output_path = os.path.join(PROCESSED_DATA_PATH, 'processed_reviews.pkl')
    reviews_df.to_pickle(output_path)
    print(f"Saved processed reviews to {output_path}")
    
    return reviews_df


def main():
    """Main function to run the full data preprocessing pipeline."""
    print("=" * 60)
    print("Wongnai QA System - Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Load all data files
    print("\n[1/4] Loading reviews...")
    reviews_df = load_reviews(REVIEW_TRAIN_FILE)
    print(f"Loaded {len(reviews_df)} reviews")
    
    print("\n[2/4] Loading food dictionary...")
    food_dict = load_food_dictionary(FOOD_DICT_FILE)
    # Filter to items 4-50 chars for efficiency (409K -> ~manageable subset)
    food_dict = [f for f in food_dict if 4 <= len(f) <= 50]
    # Limit to first 5000 unique items for speed
    food_dict = list(dict.fromkeys(food_dict))[:5000]
    food_dict_set = set(food_dict)
    print(f"Loaded {len(food_dict)} food items (filtered for efficiency)")
    
    print("\n[3/4] Loading queries...")
    queries_judges = load_queries(QUERY_JUDGES_FILE)
    queries_algo = load_queries(QUERY_ALGO_FILE)
    print(f"Loaded {len(queries_judges)} judge-labeled queries")
    print(f"Loaded {len(queries_algo)} algorithm-labeled queries")
    
    # Process reviews
    print("\n[4/4] Processing reviews...")
    processed_df = process_all_reviews(reviews_df, food_dict_set)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal reviews processed: {len(processed_df)}")
    
    print("\nRating distribution:")
    rating_dist = processed_df['star_rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"  {rating} stars: {count} ({count/len(processed_df)*100:.1f}%)")
    
    print("\nTop 10 cuisines:")
    all_cuisines = []
    for cuisines in processed_df['cuisine_type']:
        all_cuisines.extend(cuisines)
    cuisine_counts = pd.Series(all_cuisines).value_counts().head(10)
    for cuisine, count in cuisine_counts.items():
        print(f"  {cuisine}: {count}")
    
    print("\nTop 10 food types:")
    all_foods = []
    for foods in processed_df['food_type']:
        all_foods.extend(foods)
    food_counts = pd.Series(all_foods).value_counts().head(10)
    for food, count in food_counts.items():
        print(f"  {food}: {count}")
    
    print("\nTop 10 atmospheres:")
    all_atmos = []
    for atmos in processed_df['atmosphere']:
        all_atmos.extend(atmos)
    atmos_counts = pd.Series(all_atmos).value_counts().head(10)
    for atmos, count in atmos_counts.items():
        print(f"  {atmos}: {count}")
    
    print("\nTop 10 locations:")
    all_locs = []
    for locs in processed_df['location']:
        all_locs.extend(locs)
    loc_counts = pd.Series(all_locs).value_counts().head(10)
    for loc, count in loc_counts.items():
        print(f"  {loc}: {count}")
    
    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
