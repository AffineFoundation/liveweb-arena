"""CoinGecko page content extractor for GT extraction"""

import re
from typing import Any, Dict, Optional

from liveweb_arena.core.gt_extraction import PageExtractor


class CoinGeckoExtractor(PageExtractor):
    """Extract crypto data from CoinGecko pages"""

    # Mapping from URL slug to CoinGecko API coin_id
    # Some coins have different URL slugs vs API IDs
    SLUG_TO_COIN_ID = {
        # URL slugs that differ from coin_id
        "usdc": "usd-coin",
        "bnb": "binancecoin",
        "xrp": "ripple",
        "steth": "staked-ether",
        "lido-staked-eth": "staked-ether",
        "lido-staked-ether": "staked-ether",
        "avax": "avalanche-2",
        "avalanche": "avalanche-2",
        "hbar": "hedera-hashgraph",
        "hedera": "hedera-hashgraph",
        "shib": "shiba-inu",
        "dot": "polkadot",
        "ltc": "litecoin",
        "bch": "bitcoin-cash",
        "uni": "uniswap",
        "near": "near",
        "apt": "aptos",
        "icp": "internet-computer",
        "tao": "bittensor",
        "render": "render-token",
        "fet": "fetch-ai",
        "akt": "akash-network",
        "arb": "arbitrum",
        "op": "optimism",
        "pol": "polygon-ecosystem-token",
        "matic": "polygon-ecosystem-token",
        "polygon": "polygon-ecosystem-token",
        "atom": "cosmos",
        "fil": "filecoin",
        "grt": "the-graph",
        "inj": "injective-protocol",
        "injective": "injective-protocol",
        "xmr": "monero",
        "trx": "tron",
        "link": "chainlink",
        "xlm": "stellar",
        "ada": "cardano",
        "sol": "solana",
        "doge": "dogecoin",
        "usdt": "tether",
        "btc": "bitcoin",
        "eth": "ethereum",
        # Also add full names for direct mapping
        "near-protocol": "near",
        # Additional mappings for common coins
        "artificial-superintelligence-alliance": "fetch-ai",
        "asi": "fetch-ai",
        "super-alliance": "fetch-ai",
    }

    @property
    def source_name(self) -> str:
        return "coingecko"

    def matches_url(self, url: str) -> bool:
        return "coingecko.com" in url.lower()

    def classify_page(self, url: str) -> str:
        url_lower = url.lower()
        if "/coins/" in url_lower and not url_lower.endswith("/coins/"):
            return "detail"
        if "coingecko.com" in url_lower and "/coins/" not in url_lower:
            return "homepage"
        return "other"

    def extract(self, url: str, content: str) -> Dict[str, Dict[str, Any]]:
        page_type = self.classify_page(url)

        if page_type == "detail":
            return self._extract_detail(url, content)
        elif page_type == "homepage":
            return self._extract_homepage(content)
        return {}

    def _slug_to_coin_id(self, slug: str) -> str:
        """Convert URL slug to CoinGecko API coin_id."""
        slug = slug.lower()
        return self.SLUG_TO_COIN_ID.get(slug, slug)

    def _extract_detail(self, url: str, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract from coin detail page.

        CoinGecko detail pages show:
        - Current price (e.g., "$100,000.00" or "$0.00001234")
        - 24h change with sign (e.g., "+5.23%" or "-2.15%")
        - Market cap (e.g., "$1.95 Trillion")

        The accessibility tree contains structured data we can parse.
        """
        # Get coin_id from URL
        match = re.search(r'/coins/([a-zA-Z0-9_-]+)', url.lower())
        if not match:
            return {}

        slug = match.group(1)
        coin_id = self._slug_to_coin_id(slug)

        data = {}

        # Extract current price
        # Look for price patterns: "$100,000.00", "$0.00001234"
        price = self._extract_price(content)
        if price is not None:
            data['current_price'] = price

        # Extract 24h change with sign
        change = self._extract_change_24h(content)
        if change is not None:
            data['change_24h'] = change

        # Extract market cap
        market_cap = self._extract_market_cap(content)
        if market_cap is not None:
            data['market_cap'] = market_cap

        return {coin_id: data} if data else {}

    def _extract_price(self, content: str) -> Optional[float]:
        """Extract current price from page content.

        Handles various formats:
        - Large prices: $100,000.00
        - Small prices: $0.00001234
        - Very small: $0.0000000001234
        """
        # Strategy: Look for the main price display
        # Usually appears early in the page with a $ sign

        # Pattern for prices with $ sign
        price_patterns = [
            # Standard price: $100,000.00 or $1,234.56
            r'\$\s*([\d,]+\.?\d*)\s*(?:USD)?(?:\s|$)',
            # Small decimals: $0.001234
            r'\$\s*(0\.0*[1-9]\d*)\s*(?:USD)?',
            # Scientific notation style in accessibility tree
            r'(?:Price|Current)[:\s]*\$?\s*([\d,\.]+)',
        ]

        candidates = []

        for pattern in price_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for m in matches:
                try:
                    price_str = m.replace(',', '').strip()
                    if price_str:
                        price = float(price_str)
                        # Sanity check: crypto prices range from tiny to ~$100k
                        if 0 < price < 1000000:
                            candidates.append(price)
                except ValueError:
                    continue

        # Return the first reasonable price found
        # (Usually the main price appears first in the accessibility tree)
        if candidates:
            return candidates[0]
        return None

    def _extract_change_24h(self, content: str) -> Optional[float]:
        """Extract 24h price change percentage with sign.

        IMPORTANT: CoinGecko pages show multiple time-frame changes (1h, 24h, 7d, 30d).
        We must specifically target the 24h value to avoid extracting wrong data.
        """
        # Priority 1: Look for explicit "24h" context (most reliable)
        patterns_24h_explicit = [
            # "24h" or "24 hour" followed by percentage
            r'24\s*[hH](?:our)?[:\s]*([+\-]?\d+\.?\d*)\s*%',
            # "1d" or "1 day" followed by percentage
            r'1\s*[dD](?:ay)?[:\s]*([+\-]?\d+\.?\d*)\s*%',
            # Percentage immediately followed by "24h" context
            r'([+\-]?\d+\.?\d*)\s*%\s*(?:in\s+)?24\s*[hH]',
            # "past 24 hours" context
            r'(?:past|last)\s+24\s*[hH](?:ours?)?[:\s]*([+\-]?\d+\.?\d*)\s*%',
        ]

        for pattern in patterns_24h_explicit:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1).replace(' ', ''))
                    if -100 < value < 200:
                        return value
                except (ValueError, TypeError):
                    continue

        # Priority 2: Arrow indicators (usually shown prominently for main change)
        arrow_patterns = [
            (r'[▲↑]\s*(\d+\.?\d*)\s*%', 'positive'),
            (r'[▼↓]\s*(\d+\.?\d*)\s*%', 'negative'),
        ]

        for pattern, sign_hint in arrow_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    value = float(match.group(1))
                    if sign_hint == 'negative':
                        value = -value
                    if -100 < value < 200:
                        return value
                except (ValueError, TypeError):
                    continue

        # Priority 3: Signed percentage near price (first occurrence is usually 24h)
        # But ONLY if we find it near price context
        price_context = re.search(
            r'\$[\d,]+\.?\d*\s+([+\-]\d+\.?\d*)\s*%',
            content
        )
        if price_context:
            try:
                value = float(price_context.group(1))
                if -50 < value < 100:  # Tighter sanity check
                    return value
            except (ValueError, TypeError):
                pass

        return None

    def _extract_market_cap(self, content: str) -> Optional[float]:
        """Extract market cap value.

        Patterns:
        - "$1.95 Trillion" -> 1950000000000
        - "$195 Billion" -> 195000000000
        - "$1.90B" -> 1900000000
        - "$1,950,000,000,000"
        - "Market Cap $1,904,008,372"
        """
        cap_patterns = [
            # "Market Cap $1.90 billion" or "Market Cap: $1.90B"
            r'(?:Market\s*Cap|Market\s*Capitalization|MCap)[:\s]*\$?\s*([\d,\.]+)\s*(T|B|M|trillion|billion|million)?',
            # "$1.90 billion" with suffix
            r'\$\s*([\d,\.]+)\s*(trillion|billion|million|T|B|M)\b',
            # Large number with commas: "$1,904,008,372"
            r'\$\s*([\d,]{10,})',
            # Market cap followed by large number
            r'(?:Market\s*Cap)[:\s]*\$?\s*([\d,]+)',
        ]

        for pattern in cap_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for m in matches:
                try:
                    if isinstance(m, tuple):
                        value_str = m[0] if m[0] else ''
                        suffix = m[1].lower() if len(m) > 1 and m[1] else ''
                    else:
                        value_str = m
                        suffix = ''

                    value = float(value_str.replace(',', ''))

                    # Handle suffixes
                    if suffix in ('t', 'trillion'):
                        value *= 1e12
                    elif suffix in ('b', 'billion'):
                        value *= 1e9
                    elif suffix in ('m', 'million'):
                        value *= 1e6

                    # Sanity check for market cap (should be > $1M and < $10T)
                    if 1e6 < value < 1e13:
                        return value
                except (ValueError, AttributeError, TypeError):
                    continue

        return None

    def _extract_homepage(self, content: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract from homepage - marks data as unsigned since homepage
        may not show explicit +/- signs (uses colors instead).
        """
        result = {}

        # Homepage shows table rows with coin data
        # Multiple patterns to handle different accessibility tree formats
        patterns = [
            # Pattern 1: "Bitcoin BTC $100,000 5.2%"
            re.compile(
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+'  # Coin name
                r'([A-Z]{2,10})\s+'                      # Symbol
                r'\$?\s*([\d,]+\.?\d*)\s*'               # Price
                r'([+\-]?\d+\.?\d*)\s*%',                # Change
                re.IGNORECASE
            ),
            # Pattern 2: Name followed by price and change (no symbol)
            re.compile(
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+'  # Coin name
                r'\$\s*([\d,]+\.?\d*)\s+'                # Price
                r'([+\-]?\d+\.?\d*)\s*%',                # Change
                re.IGNORECASE
            ),
            # Pattern 3: Handle multiword names with flexible spacing
            re.compile(
                r'((?:Internet|Bitcoin|Binance|Shiba|Lido|Near|Polygon)\s+\w+)\s+'
                r'\$?\s*([\d,]+\.?\d*)\s*'
                r'([+\-]?\d+\.?\d*)\s*%',
                re.IGNORECASE
            ),
        ]

        for pattern in patterns:
            for match in pattern.finditer(content):
                try:
                    groups = match.groups()
                    if len(groups) >= 3:
                        name = groups[0].strip().lower().replace(' ', '-')
                        # Handle patterns with/without symbol
                        if len(groups) == 4:
                            price = float(groups[2].replace(',', ''))
                            change_str = groups[3]
                        else:
                            price = float(groups[1].replace(',', ''))
                            change_str = groups[2]

                        change = float(change_str)
                        has_sign = change_str.startswith('+') or change_str.startswith('-')

                        coin_id = self._name_to_id(name)
                        if coin_id and coin_id not in result:
                            result[coin_id] = {
                                'current_price': price,
                                'change_24h': change,
                                'change_unsigned': not has_sign,
                            }
                except (ValueError, IndexError):
                    pass

        return result

    def _name_to_id(self, name: str) -> Optional[str]:
        """Map coin name to CoinGecko ID"""
        # Comprehensive mapping covering all cached coins and hybrid template assets
        name_map = {
            # Major coins
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'tether': 'tether',
            'xrp': 'ripple',
            'ripple': 'ripple',
            'solana': 'solana',
            'bnb': 'binancecoin',
            'binance coin': 'binancecoin',
            'dogecoin': 'dogecoin',
            'cardano': 'cardano',
            # Additional coins used in hybrid templates
            'internet-computer': 'internet-computer',
            'internet computer': 'internet-computer',
            'icp': 'internet-computer',
            'polkadot': 'polkadot',
            'chainlink': 'chainlink',
            'litecoin': 'litecoin',
            'uniswap': 'uniswap',
            'stellar': 'stellar',
            'cosmos': 'cosmos',
            'near': 'near',
            'near protocol': 'near',
            'aptos': 'aptos',
            'sui': 'sui',
            'tao': 'bittensor',
            'bittensor': 'bittensor',
            'avalanche': 'avalanche-2',
            'tron': 'tron',
            'bitcoin cash': 'bitcoin-cash',
            'bitcoin-cash': 'bitcoin-cash',
            'filecoin': 'filecoin',
            'hedera': 'hedera',
            'hedera hashgraph': 'hedera',
        }
        return name_map.get(name.lower())
