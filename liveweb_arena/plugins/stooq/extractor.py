"""Stooq page content extractor for GT extraction"""

import re
from typing import Any, Dict
from urllib.parse import urlparse, parse_qs

from liveweb_arena.core.gt_extraction import PageExtractor


class StooqExtractor(PageExtractor):
    """Extract financial data from Stooq pages"""

    @property
    def source_name(self) -> str:
        return "stooq"

    def matches_url(self, url: str) -> bool:
        return "stooq.com" in url.lower()

    def classify_page(self, url: str) -> str:
        """Classify page type based on URL.

        Stooq has quote pages with symbol parameter:
        - https://stooq.com/q/?s=aapl.us (detail page for specific symbol)
        - https://stooq.com/ (homepage)
        """
        parsed = urlparse(url)
        query = parse_qs(parsed.query)

        if "s" in query and query["s"]:
            return "detail"
        if parsed.path == "/" or parsed.path == "":
            return "homepage"
        return "other"

    def extract(self, url: str, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract price data from Stooq page accessibility tree."""
        page_type = self.classify_page(url)

        if page_type == "detail":
            return self._extract_detail(url, content)
        elif page_type == "homepage":
            return self._extract_homepage(content)
        return {}

    def _extract_symbol(self, url: str) -> str:
        """Extract symbol from URL query parameter."""
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if "s" in query and query["s"]:
            return query["s"][0].lower()
        return ""

    def _extract_detail(self, url: str, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract data from symbol detail page.

        Stooq detail pages show:
        - Current/Last price
        - Daily change (absolute and percentage)
        - Open, High, Low prices
        - Volume

        The accessibility tree typically contains lines like:
        - "Last 255.53"
        - "Change +2.31 (+0.91%)"
        - "Open 254.00"
        - "High 256.10"
        - "Low 253.80"
        """
        symbol = self._extract_symbol(url)
        if not symbol:
            return {}

        data = {}

        # Extract last/current price
        # Look for patterns like "Last $ 258.270" or "Kurs 255.53" (Polish)
        # The accessibility tree may have:
        # - "Last $ 258.270" or "Last\n$ 258.270" (with newline)
        # - "258.270 +2.86 (+1.12%)" format for inline display
        last_patterns = [
            # "Last $ 258.270" or "Last\n$ 258.270"
            r'Last[\s\n]*\$?\s*([\d,\.]+)',
            # Polish: "Kurs 255.53"
            r'(?:Kurs|Ostatni)\s*[:\s]*([\d,\.]+)',
            # Inline format in header: "258.270  +2.860 (+1.12%)"
            r'([\d,\.]+)\s+[+\-][\d,\.]+\s+\([+\-][\d,\.]+%\)',
        ]
        for pattern in last_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    price_str = match.group(1).replace(',', '').replace(' ', '')
                    price = float(price_str)
                    # Filter out percentages (typically < 100) vs prices
                    if price > 0.1:  # Most stock prices are > $0.10
                        data['last_price'] = price
                        data['current_price'] = price  # Alias
                        break
                except ValueError:
                    continue

        # Extract daily change with sign
        # Patterns: "+2.31 (+0.91%)", "Change: -1.5%", "-0.91%"
        # First try to find both absolute and percentage together
        both_match = re.search(
            r'([+\-]?\d+\.?\d*)\s*\(([+\-]?\d+\.?\d*)\s*%\)',
            content
        )
        if both_match:
            try:
                data['daily_change'] = float(both_match.group(1))
                data['daily_change_pct'] = float(both_match.group(2))
            except ValueError:
                pass

        # If we didn't get percentage, try standalone percentage patterns
        if 'daily_change_pct' not in data:
            pct_patterns = [
                # Percentage change alone: "+0.91%" or "-0.91%"
                r'(?:Change|Zmiana)[:\s]*([+\-]?\d+\.?\d*)\s*%',
                # Signed percentage in text (common format)
                r'([+\-]\d+\.?\d*)\s*%',
                # Unsigned percentage near a sign indicator
                r'(\d+\.?\d*)\s*%',
            ]
            for pattern in pct_patterns:
                match = re.search(pattern, content)
                if match:
                    try:
                        pct = float(match.group(1))
                        # Sanity check: daily change is typically -50% to +50%
                        if -100 < pct < 100:
                            data['daily_change_pct'] = pct
                            break
                    except ValueError:
                        continue

        # If we have absolute change but not percentage, calculate from price
        if 'daily_change' in data and 'daily_change_pct' not in data:
            if 'last_price' in data and data['last_price'] > 0:
                prev_price = data['last_price'] - data['daily_change']
                if prev_price > 0:
                    data['daily_change_pct'] = (data['daily_change'] / prev_price) * 100

        # Extract open price
        open_match = re.search(r'(?:Open|Otwarcie)[:\s]*([\d,\.]+)', content, re.IGNORECASE)
        if open_match:
            try:
                data['open'] = float(open_match.group(1).replace(',', ''))
            except ValueError:
                pass

        # Extract high price
        high_match = re.search(r'(?:High|Max|Maks)[:\s]*([\d,\.]+)', content, re.IGNORECASE)
        if high_match:
            try:
                data['high'] = float(high_match.group(1).replace(',', ''))
            except ValueError:
                pass

        # Extract low price
        low_match = re.search(r'(?:Low|Min)[:\s]*([\d,\.]+)', content, re.IGNORECASE)
        if low_match:
            try:
                data['low'] = float(low_match.group(1).replace(',', ''))
            except ValueError:
                pass

        # Extract volume
        vol_match = re.search(r'(?:Volume|Vol|Wolumen)[:\s]*([\d,\.]+[KMB]?)', content, re.IGNORECASE)
        if vol_match:
            try:
                vol_str = vol_match.group(1).replace(',', '')
                multiplier = 1
                if vol_str.endswith('K'):
                    multiplier = 1000
                    vol_str = vol_str[:-1]
                elif vol_str.endswith('M'):
                    multiplier = 1000000
                    vol_str = vol_str[:-1]
                elif vol_str.endswith('B'):
                    multiplier = 1000000000
                    vol_str = vol_str[:-1]
                data['volume'] = float(vol_str) * multiplier
            except ValueError:
                pass

        return {symbol: data} if data else {}

    def _extract_homepage(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract data from Stooq homepage.

        Homepage may show market overview tables with indices, currencies, etc.
        This is less reliable than detail pages, so we mark certain data as unsigned.
        """
        result = {}

        # Try to extract major index values from homepage tables
        # Pattern: "^DJI  42,583.32  +0.35%"
        index_pattern = re.compile(
            r'\^(\w+)\s+'           # Index symbol like ^DJI
            r'([\d,\.]+)\s+'        # Value
            r'([+\-]?\d+\.?\d*)\s*%',  # Percentage change
            re.IGNORECASE
        )

        for match in index_pattern.finditer(content):
            try:
                symbol = f"^{match.group(1).lower()}"
                value = float(match.group(2).replace(',', ''))
                change_pct = float(match.group(3))

                result[symbol] = {
                    'last_price': value,
                    'current_price': value,
                    'daily_change_pct': change_pct,
                }
            except (ValueError, IndexError):
                continue

        return result
