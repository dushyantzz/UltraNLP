import re
import html
import unicodedata
from typing import List, Dict, Set, Optional, Tuple, Union
from collections import defaultdict, Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from enum import Enum
import array

# Token type enumeration for memory efficiency
class TokenType(Enum):
    WORD = 1
    NUMBER = 2
    EMAIL = 3
    URL = 4
    CURRENCY = 5
    PHONE = 6
    HASHTAG = 7
    MENTION = 8
    EMOJI = 9
    PUNCTUATION = 10
    WHITESPACE = 11
    DATETIME = 12
    CONTRACTION = 13
    HYPHENATED = 14

@dataclass(slots=True)  # Memory optimization
class Token:
    text: str
    start: int
    end: int
    token_type: TokenType
    
    def __hash__(self):
        return hash((self.text, self.token_type))

class UltraFastTokenizer:
    """Ultra-optimized tokenizer using compiled patterns and efficient algorithms"""
    
    def __init__(self):
        # Pre-compiled patterns with flags for maximum speed
        self._patterns = self._compile_patterns()
        self._unicode_categories = self._build_unicode_cache()
        
        # Thread-local storage for better performance in multi-threaded environments
        self._local = threading.local()
        
    def _compile_patterns(self) -> List[Tuple[re.Pattern, TokenType]]:
        """Compile all regex patterns with optimizations"""
        patterns = [
            # Email (optimized pattern)
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE), TokenType.EMAIL),
            
            # URLs (comprehensive but fast)
            (re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE), TokenType.URL),
            
            # Currency patterns (optimized for speed)
            (re.compile(r'(?:\$|€|£|¥|₹|₽)\d+(?:\.\d{1,4})?(?:[KMBkmb])?|\d+(?:\.\d{1,4})?(?:USD|EUR|GBP|JPY|INR|RUB|Rs)\b', re.IGNORECASE), TokenType.CURRENCY),
            
            # Phone numbers (efficient pattern)
            (re.compile(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'), TokenType.PHONE),
            
            # Date/Time patterns (optimized)
            (re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s?[AaPp][Mm])?)\b'), TokenType.DATETIME),
            
            # Social media (hashtags and mentions)
            (re.compile(r'#\w+'), TokenType.HASHTAG),
            (re.compile(r'@\w+'), TokenType.MENTION),
            
            # Contractions and hyphenated words
            (re.compile(r"\b\w+(?:'\w+)+\b"), TokenType.CONTRACTION),
            (re.compile(r'\b\w+(?:-\w+)+\b'), TokenType.HYPHENATED),
            
            # Emojis (optimized Unicode ranges)
            (re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF]+'), TokenType.EMOJI),
            
            # Numbers (including decimals and scientific notation)
            (re.compile(r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b'), TokenType.NUMBER),
            
            # Words (Unicode-aware)
            (re.compile(r'\b\w+\b'), TokenType.WORD),
            
            # Punctuation
            (re.compile(r'[^\w\s]'), TokenType.PUNCTUATION),
        ]
        return patterns
    
    @lru_cache(maxsize=10000)
    def _build_unicode_cache(self) -> Dict[str, str]:
        """Build Unicode category cache for faster processing"""
        return {}
    
    def tokenize_fast(self, text: str) -> List[Token]:
        """Ultra-fast tokenization using optimized algorithms"""
        if not text:
            return []
        
        tokens = []
        text_len = len(text)
        processed = [False] * text_len  # Boolean array for tracking processed positions
        
        # Use compiled patterns in order of specificity
        for pattern, token_type in self._patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                
                # Skip if any position in range is already processed
                if any(processed[i] for i in range(start, end)):
                    continue
                
                # Mark positions as processed
                for i in range(start, end):
                    processed[i] = True
                
                tokens.append(Token(
                    text=match.group(),
                    start=start,
                    end=end,
                    token_type=token_type
                ))
        
        # Sort by position (using key function for speed)
        tokens.sort(key=lambda x: x.start)
        return tokens
    
    def batch_tokenize(self, texts: List[str], max_workers: int = 4) -> List[List[Token]]:
        """Parallel batch tokenization for maximum throughput"""
        if len(texts) < 100:  # Use single thread for small batches
            return [self.tokenize_fast(text) for text in texts]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.tokenize_fast, texts))

class HyperSpeedCleaner:
    """Hyper-optimized text cleaner using compiled patterns and efficient algorithms"""
    
    def __init__(self):
        # Pre-compile all patterns for maximum speed
        self._url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+', 
            re.IGNORECASE | re.MULTILINE
        )
        self._email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
            re.IGNORECASE
        )
        self._phone_pattern = re.compile(
            r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        )
        self._emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
            r'\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF'
            r'\U0001F018-\U0001F270\U00002700-\U000027bf]+'
        )
        self._whitespace_pattern = re.compile(r'\s+')
        self._html_pattern = re.compile(r'<[^>]+>')
        self._special_chars_pattern = re.compile(r'[^\w\s.,!?;:-]')
        
        # HTML entity mapping for fast decoding
        self._html_entities = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
            '&apos;': "'", '&nbsp;': ' ', '&#39;': "'", '&#x27;': "'",
            '&#x2F;': '/', '&#x60;': '`', '&#x3D;': '='
        }
    
    @lru_cache(maxsize=50000)
    def _clean_cached(self, text: str, options_hash: int) -> str:
        """Cached cleaning for repeated text patterns"""
        return self._clean_internal(text, options_hash)
    
    def _clean_internal(self, text: str, options_hash: int) -> str:
        """Internal cleaning method"""
        # Decode options from hash (simple bit manipulation)
        lowercase = bool(options_hash & 1)
        remove_html = bool(options_hash & 2)
        remove_urls = bool(options_hash & 4)
        remove_emails = bool(options_hash & 8)
        remove_phones = bool(options_hash & 16)
        remove_emojis = bool(options_hash & 32)
        normalize_whitespace = bool(options_hash & 64)
        remove_special_chars = bool(options_hash & 128)
        
        # Fast HTML entity decoding
        for entity, replacement in self._html_entities.items():
            if entity in text:
                text = text.replace(entity, replacement)
        
        # Remove HTML tags
        if remove_html:
            text = self._html_pattern.sub(' ', text)
        
        # Remove URLs
        if remove_urls:
            text = self._url_pattern.sub('', text)
        
        # Remove emails
        if remove_emails:
            text = self._email_pattern.sub('', text)
        
        # Remove phone numbers
        if remove_phones:
            text = self._phone_pattern.sub('', text)
        
        # Remove emojis
        if remove_emojis:
            text = self._emoji_pattern.sub('', text)
        
        # Remove special characters
        if remove_special_chars:
            text = self._special_chars_pattern.sub('', text)
        
        # Normalize whitespace
        if normalize_whitespace:
            text = self._whitespace_pattern.sub(' ', text).strip()
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        return text
    
    def clean(self, text: str, options: Optional[Dict] = None) -> str:
        """Ultra-fast text cleaning with caching"""
        if not text:
            return ""
        
        if options is None:
            options = {
                'lowercase': True,
                'remove_html': True,
                'remove_urls': True,
                'remove_emails': False,
                'remove_phones': False,
                'remove_emojis': True,
                'normalize_whitespace': True,
                'remove_special_chars': False
            }
        
        # Create hash from options for caching
        options_hash = (
            int(options.get('lowercase', True)) |
            (int(options.get('remove_html', True)) << 1) |
            (int(options.get('remove_urls', True)) << 2) |
            (int(options.get('remove_emails', False)) << 3) |
            (int(options.get('remove_phones', False)) << 4) |
            (int(options.get('remove_emojis', True)) << 5) |
            (int(options.get('normalize_whitespace', True)) << 6) |
            (int(options.get('remove_special_chars', False)) << 7)
        )
        
        return self._clean_cached(text, options_hash)
    
    def batch_clean(self, texts: List[str], options: Optional[Dict] = None, max_workers: int = 4) -> List[str]:
        """Parallel batch cleaning for maximum throughput"""
        if len(texts) < 50:  # Use single thread for small batches
            return [self.clean(text, options) for text in texts]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda text: self.clean(text, options), texts))

class LightningSpellCorrector:
    """Lightning-fast spell corrector with advanced caching and algorithms"""
    
    def __init__(self):
        self.word_freq = Counter()
        self._correction_cache = {}
        self._edit_distance_cache = {}
        self.alphabet = set('abcdefghijklmnopqrstuvwxyz')
        self._lock = threading.Lock()
        
        # Pre-load common English words for immediate use
        self._load_common_words()
    
    def _load_common_words(self):
        """Load most common English words"""
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
        ]
        
        for word in common_words:
            self.word_freq[word] = 1000  # High frequency for common words
    
    @lru_cache(maxsize=10000)
    def _edits1(self, word: str) -> frozenset:
        """Generate edit distance 1 words with caching"""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = frozenset(L + R[1:] for L, R in splits if R)
        transposes = frozenset(L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1)
        replaces = frozenset(L + c + R[1:] for L, R in splits if R for c in self.alphabet)
        inserts = frozenset(L + c + R for L, R in splits for c in self.alphabet)
        return deletes | transposes | replaces | inserts
    
    def correct(self, word: str) -> str:
        """Ultra-fast spell correction with multi-level caching"""
        word_lower = word.lower()
        
        # Level 1: Direct cache lookup
        if word_lower in self._correction_cache:
            return self._correction_cache[word_lower]
        
        # Level 2: Word frequency lookup (already correct)
        if word_lower in self.word_freq:
            self._correction_cache[word_lower] = word
            return word
        
        # Level 3: Edit distance 1
        candidates = self._edits1(word_lower) & set(self.word_freq.keys())
        if candidates:
            best = max(candidates, key=self.word_freq.get)
            self._correction_cache[word_lower] = best
            return best
        
        # Level 4: Edit distance 2 (limited to prevent slowdown)
        candidates = set()
        for edit1 in self._edits1(word_lower):
            if len(candidates) > 100:  # Limit to prevent exponential growth
                break
            candidates.update(self._edits1(edit1) & set(self.word_freq.keys()))
        
        if candidates:
            best = max(candidates, key=self.word_freq.get)
            self._correction_cache[word_lower] = best
            return best
        
        # Return original if no correction found
        self._correction_cache[word_lower] = word
        return word
    
    def train(self, text: str):
        """Train spell corrector on text corpus"""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        with self._lock:
            self.word_freq.update(words)

class UltraNLPProcessor:
    """Main processor class with optimized pipeline"""
    
    def __init__(self):
        self.tokenizer = UltraFastTokenizer()
        self.cleaner = HyperSpeedCleaner()
        self.spell_corrector = LightningSpellCorrector()
        
        # Performance monitoring
        self._stats = {
            'documents_processed': 0,
            'total_tokens': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def process(self, text: str, options: Optional[Dict] = None) -> Dict:
        """Lightning-fast complete NLP preprocessing"""
        if not text:
            return {'tokens': [], 'cleaned_text': '', 'original_text': text}
        
        options = options or {}
        
        # Step 1: Clean text (with caching)
        cleaned_text = self.cleaner.clean(text, options.get('clean_options'))
        
        # Step 2: Tokenize (ultra-fast)
        tokens = self.tokenizer.tokenize_fast(cleaned_text)
        
        # Step 3: Spell correction (optional, for WORD tokens only)
        if options.get('spell_correct', False):
            for token in tokens:
                if token.token_type == TokenType.WORD:
                    token.text = self.spell_corrector.correct(token.text)
        
        # Update stats
        self._stats['documents_processed'] += 1
        self._stats['total_tokens'] += len(tokens)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': [token.text for token in tokens],
            'token_objects': tokens,
            'token_count': len(tokens),
            'processing_stats': self._stats.copy()
        }
    
    def batch_process(self, texts: List[str], options: Optional[Dict] = None, max_workers: int = 4) -> List[Dict]:
        """Ultra-fast batch processing with parallel execution"""
        if len(texts) < 20:  # Single-threaded for small batches
            return [self.process(text, options) for text in texts]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda text: self.process(text, options), texts))
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self._stats.copy()

# Example benchmark and usage
if __name__ == "__main__":
    import time
    
    # Initialize processor
    processor = UltraNLPProcessor()
    
    # Sample test data
    test_texts = [
        "Hey there! 😊 Check out https://example.com for $20.99 deals!",
        "Contact support@company.com or call +1-555-123-4567 #urgent",
        "<b>HTML content</b> with misspelld words and www.test.com",
        "Meeting at 2:30PM on 12/25/2024. Don't be late! @everyone",
        "Price: ₹1,500.50K for premium subscription 💰"
    ] * 100  # Multiply for batch testing
    
    # Benchmark
    start_time = time.time()
    results = processor.batch_process(test_texts, {
        'clean_options': {
            'remove_urls': True,
            'remove_emojis': True,
            'lowercase': True
        },
        'spell_correct': False  # Disabled for speed
    })
    end_time = time.time()
    
    print(f"Processed {len(test_texts)} documents in {end_time - start_time:.4f} seconds")
    print(f"Speed: {len(test_texts) / (end_time - start_time):.2f} documents/second")
    print(f"Performance stats: {processor.get_performance_stats()}")
    
    # Show sample result
    print("\nSample result:")
    print(f"Original: {results[0]['original_text']}")
    print(f"Cleaned: {results[0]['cleaned_text']}")
    print(f"Tokens: {results[0]['tokens'][:10]}...")
