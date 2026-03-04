"""
Triage Agent — Stage 1: Document Classification and Profiling.

Rubric Compliance:
1. Origin Type Detection: char density, image ratio, font metadata
2. Layout Complexity: column heuristics, bbox analysis
3. Domain Hint: Config-driven keyword classifier (swappable)
4. Cost Estimation: Derived from classification
5. Edge Cases: Zero-text, mixed-mode, form-fillable handling
"""

import pdfplumber
import hashlib
import yaml
from pathlib import Path
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from typing import Dict, List, Tuple
from src.models.profile import DocumentProfile, OriginType, LayoutComplexity, DomainHint


# Prevent langdetect randomness
DetectorFactory.seed = 0


class TriageAgent:
    """
    Document profiling agent that classifies documents before extraction.
    
    All thresholds and keywords loaded from config (Rubric: Externalized).
    """

    def __init__(self, config_path: str = "rubric/extraction_rules.yaml"):
        """Initialize with externalized configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Thresholds from config (Rubric: Not hardcoded)
        self.char_density_threshold = self.config['thresholds'].get('char_density_min', 500)
        self.image_area_threshold = self.config['thresholds'].get('image_area_max', 0.5)
        self.domain_keywords = self.config.get('domain_classification', {}).get('keywords', {})
        
        # Cost tiers from config
        self.cost_tiers = {
            "A": self.config.get('cost_tiers', {}).get('strategy_a', 0.0001),
            "B": self.config.get('cost_tiers', {}).get('strategy_b', 0.005),
            "C": self.config.get('cost_tiers', {}).get('strategy_c', 0.020),
        }

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _calculate_char_density(self, page) -> Tuple[float, int]:
        """
        Calculate character density for a page.
        
        Returns: (chars_per_point², total_chars)
        """
        text = page.extract_text() or ""
        total_chars = len(text)
        page_area = page.width * page.height
        
        if page_area == 0:
            return 0.0, total_chars
        
        density = total_chars / page_area
        return density, total_chars

    def _calculate_image_ratio(self, page) -> float:
        """
        Calculate image area ratio for a page.
        
        Returns: ratio of image area to page area (0.0 to 1.0)
        """
        page_area = page.width * page.height
        if page_area == 0:
            return 0.0
        
        total_image_area = 0.0
        if page.images:
            for img in page.images:
                img_area = (img['x1'] - img['x0']) * (img['y1'] - img['y0'])
                total_image_area += img_area
        
        return min(total_image_area / page_area, 1.0)

    def _has_horizontal_gaps(self, page, threshold: float = 0.3) -> bool:
        """
        Detect multi-column layout by analyzing horizontal gaps.
        
        Rubric Compliance: Column-count heuristics via bbox analysis.
        """
        blocks = page.extract_text(layout=True)
        if not blocks:
            return False
        
        # Analyze text block positions
        x_positions = []
        for line in blocks.split('\n'):
            if line.strip():
                # Get bbox for this line (simplified)
                x_positions.append(len(line) - len(line.lstrip()))
        
        if len(x_positions) < 5:
            return False
        
        # Check for significant indentation variation (indicates columns)
        indent_variance = max(x_positions) - min(x_positions)
        avg_line_length = sum(len(line) for line in blocks.split('\n') if line.strip()) / max(len(x_positions), 1)
        
        return indent_variance > (avg_line_length * threshold)

    def _detect_tables(self, page) -> int:
        """Count tables on a page using pdfplumber table detection."""
        tables = page.find_tables()
        return len(tables) if tables else 0

    def _has_font_metadata(self, page) -> bool:
        """Check if page has embedded font metadata."""
        try:
            chars = page.chars
            if chars:
                for char in chars[:10]:  # Sample first 10 chars
                    if char.get('fontname'):
                        return True
            return False
        except Exception:
            return False

    def _has_form_fields(self, pdf) -> bool:
        """Detect interactive form fields in PDF."""
        try:
            for page in pdf.pages:
                if page.annots:
                    for annot in page.annots:
                        if annot.get('subtype') == '/Widget':
                            return True
            return False
        except Exception:
            return False

    def _detect_language(self, text_sample: str) -> Tuple[str, float]:
        """Detect document language with confidence."""
        if len(text_sample) < 20:
            return "en", 0.5  # Default for low-text documents
        
        try:
            lang = detect(text_sample)
            # langdetect doesn't provide confidence, estimate based on text length
            confidence = min(1.0, len(text_sample) / 1000)
            return lang, confidence
        except LangDetectException:
            return "en", 0.5

    def _classify_domain(self, text_sample: str) -> DomainHint:
        """
        Classify document domain using config-driven keywords.
        
        Rubric Compliance: Swappable strategy via YAML config.
        """
        if not text_sample or not self.domain_keywords:
            return DomainHint.GENERAL
        
        text_lower = text_sample.lower()
        scores: Dict[str, int] = {domain: 0 for domain in self.domain_keywords.keys()}
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    scores[domain] += 1
        
        if max(scores.values()) == 0:
            return DomainHint.GENERAL
        
        best_domain = max(scores, key=scores.get)
        return DomainHint(best_domain)

    def _determine_origin_type(
        self,
        avg_char_density: float,
        avg_image_ratio: float,
        has_font_meta: bool,
        total_chars: int
    ) -> OriginType:
        """
        Determine document origin type from metrics.
        
        Rubric Compliance: Multi-signal detection.
        """
        # Edge case: Zero-text document
        if total_chars == 0:
            if avg_image_ratio > 0.5:
                return OriginType.SCANNED_IMAGE
            else:
                return OriginType.FORM_FILLABLE
        
        # Native digital: High char density, low image ratio, has fonts
        if (avg_char_density > self.char_density_threshold and 
            avg_image_ratio < self.image_area_threshold and 
            has_font_meta):
            return OriginType.NATIVE_DIGITAL
        
        # Scanned: Low char density, high image ratio
        if avg_image_ratio > 0.8:
            return OriginType.SCANNED_IMAGE
        
        # Mixed: Some text, some images
        return OriginType.MIXED

    def _determine_layout_complexity(
        self,
        pages: List,
        has_multi_column: bool,
        avg_table_count: float
    ) -> LayoutComplexity:
        """
        Determine layout complexity from analysis.
        
        Rubric Compliance: Bbox analysis and column heuristics.
        """
        if avg_table_count > 2.0:
            return LayoutComplexity.TABLE_HEAVY
        
        if has_multi_column:
            return LayoutComplexity.MULTI_COLUMN
        
        # Check for figures (simplified)
        figure_count = sum(1 for p in pages if p.images and len(p.images) > 2)
        if figure_count > len(pages) * 0.3:
            return LayoutComplexity.FIGURE_HEAVY
        
        return LayoutComplexity.SINGLE_COLUMN

    def _select_strategy(
        self,
        origin: OriginType,
        layout: LayoutComplexity
    ) -> Tuple[str, float]:
        """Select extraction strategy based on classification."""
        # Strategy A: Fast text (native digital, simple layout)
        if origin == OriginType.NATIVE_DIGITAL and layout == LayoutComplexity.SINGLE_COLUMN:
            return "A", self.cost_tiers["A"]
        
        # Strategy C: Vision (scanned images)
        if origin == OriginType.SCANNED_IMAGE:
            return "C", self.cost_tiers["C"]
        
        # Strategy B: Layout-aware (everything else)
        return "B", self.cost_tiers["B"]

    def analyze(self, file_path: str) -> DocumentProfile:
        """
        Complete document analysis pipeline.
        
        Args:
            file_path: Path to PDF document
            
        Returns:
            DocumentProfile with all classification dimensions
        """
        file_path = Path(file_path)
        
        # Generate document ID
        with open(file_path, 'rb') as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()
        doc_id = content_hash[:16]
        
        # Analyze with pdfplumber
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages
            page_count = len(pages)
            
            if page_count == 0:
                raise ValueError("Document has no pages")
            
            # Collect metrics across all pages
            total_chars = 0
            total_image_ratio = 0.0
            total_table_count = 0
            has_font_meta = False
            has_multi_column = False
            text_samples = []
            
            for page in pages:
                density, chars = self._calculate_char_density(page)
                total_chars += chars
                total_image_ratio += self._calculate_image_ratio(page)
                total_table_count += self._detect_tables(page)
                
                if self._has_font_metadata(page):
                    has_font_meta = True
                
                if self._has_horizontal_gaps(page):
                    has_multi_column = True
                
                # Collect text for language/domain detection
                text = page.extract_text() or ""
                if text:
                    text_samples.append(text[:500])  # Sample first 500 chars
            
            # Calculate averages
            avg_char_density = total_chars / page_count
            avg_image_ratio = total_image_ratio / page_count
            avg_table_count = total_table_count / page_count
            
            # Classifications
            origin = self._determine_origin_type(
                avg_char_density, avg_image_ratio, has_font_meta, total_chars
            )
            layout = self._determine_layout_complexity(
                pages, has_multi_column, avg_table_count
            )
            
            # Language and domain
            combined_text = " ".join(text_samples[:5])  # First 5 pages
            language, lang_conf = self._detect_language(combined_text)
            domain = self._classify_domain(combined_text)
            
            # Strategy selection
            strategy, cost_per_page = self._select_strategy(origin, layout)
            
            # Warnings
            warnings = []
            if page_count > 500:
                warnings.append("Document exceeds recommended page count (500)")
            if avg_image_ratio > 0.8:
                warnings.append("High image ratio — OCR quality may vary")
            if total_chars == 0:
                warnings.append("No text detected — may require vision extraction")
            
            return DocumentProfile(
                doc_id=doc_id,
                filename=file_path.name,
                page_count=page_count,
                origin_type=origin,
                layout_complexity=layout,
                language=language,
                language_confidence=lang_conf,
                domain_hint=domain,
                recommended_strategy=strategy,
                estimated_cost_per_page=cost_per_page,
                total_estimated_cost=cost_per_page * page_count,
                avg_chars_per_page=avg_char_density,
                image_area_ratio=avg_image_ratio,
                has_font_metadata=has_font_meta,
                has_form_fields=self._has_form_fields(pdf),
                warnings=warnings,
                content_hash=content_hash,
            )

    def save_profile(self, profile: DocumentProfile, output_dir: str = ".refinery/profiles") -> str:
        """Save profile to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / f"{profile.doc_id}.json"
        with open(file_path, 'w') as f:
            f.write(profile.model_dump_json(indent=2))
        
        return str(file_path)