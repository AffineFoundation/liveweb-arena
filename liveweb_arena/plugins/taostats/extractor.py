"""Taostats page content extractor for GT extraction"""

import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from liveweb_arena.core.gt_extraction import PageExtractor


class TaostatsExtractor(PageExtractor):
    """Extract Bittensor network data from taostats.io pages"""

    @property
    def source_name(self) -> str:
        return "taostats"

    def matches_url(self, url: str) -> bool:
        return "taostats.io" in url.lower()

    def classify_page(self, url: str) -> str:
        """Classify page type.

        Taostats pages:
        - https://taostats.io/ (homepage with network overview)
        - https://taostats.io/subnets (subnet list)
        - https://taostats.io/subnets/27 (subnet detail)
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        # Subnet detail page
        if re.match(r'subnets/\d+', path):
            return "detail"
        # Subnet list page
        if path == "subnets":
            return "search"
        # Homepage
        if not path:
            return "homepage"
        return "other"

    def extract(self, url: str, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract data from taostats.io page accessibility tree."""
        page_type = self.classify_page(url)

        if page_type == "detail":
            return self._extract_subnet_detail(url, content)
        elif page_type == "homepage":
            return self._extract_homepage(content)
        elif page_type == "search":
            return self._extract_subnet_list(content)
        return {}

    def _extract_subnet_id(self, url: str) -> str:
        """Extract subnet ID from URL path."""
        parsed = urlparse(url)
        path = parsed.path

        match = re.search(r'/subnets/(\d+)', path)
        if match:
            return match.group(1)
        return ""

    def _extract_subnet_detail(self, url: str, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract data from subnet detail page.

        Subnet pages show:
        - Subnet name
        - Owner address (truncated on UI, full in data)
        - Alpha token price in TAO
        - Network statistics (emission, registration cost, etc.)
        """
        subnet_id = self._extract_subnet_id(url)
        if not subnet_id:
            return {}

        data = {}

        # Extract subnet name
        # Patterns: "Subnet 27: Compute" or "Compute" header
        name_patterns = [
            r'Subnet\s*\d+[:\s]+([A-Za-z0-9\s\-_]+?)(?:\s*\||\s*-|\n|$)',
            r'(?:Subnet Name|Name)[:\s]*([A-Za-z0-9\s\-_]+)',
        ]
        for pattern in name_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if name and len(name) > 1:
                    data['name'] = name
                    data['subnet_name'] = name
                    break

        # Extract owner address
        # Bittensor addresses start with '5' and are typically 48 chars
        owner_patterns = [
            r'(?:Owner|Owned by)[:\s]*(5[A-Za-z0-9]{47})',
            r'(?:Owner|Owned by)[:\s]*(5[A-Za-z0-9]{3,10}\.{3}[A-Za-z0-9]{3,10})',  # Truncated
        ]
        for pattern in owner_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                data['owner'] = match.group(1)
                break

        # Extract TAO price (shown on homepage/network pages)
        tao_patterns = [
            r'(?:TAO|Bittensor)\s*(?:Price)?[:\s]*\$?([\\d,\\.]+)',
            r'\$\s*([\d,\\.]+)\s*(?:per TAO|TAO|/TAO)',
        ]
        for pattern in tao_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1).replace(',', ''))
                    if 10 < price < 10000:  # Sanity check for TAO price range
                        data['tao_price'] = price
                        break
                except ValueError:
                    continue

        # Extract alpha token price (subnet-specific token)
        alpha_patterns = [
            r'(?:Alpha|Token)\s*(?:Price)?[:\s]*([\d,\\.]+)\s*(?:τ|TAO)',
            r'([\d,\\.]+)\s*(?:τ|TAO)\s*(?:per alpha|/alpha)',
        ]
        for pattern in alpha_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    data['alpha_price'] = float(match.group(1).replace(',', ''))
                    data['price'] = data['alpha_price']
                    break
                except ValueError:
                    continue

        # Extract emission
        emission_match = re.search(r'(?:Emission)[:\s]*([\d,\\.]+)\s*%?', content, re.IGNORECASE)
        if emission_match:
            try:
                data['emission'] = float(emission_match.group(1).replace(',', ''))
            except ValueError:
                pass

        # Extract registration cost
        reg_cost_match = re.search(r'(?:Registration|Reg\.?\s*Cost)[:\s]*([\d,\\.]+)\s*(?:τ|TAO)?', content, re.IGNORECASE)
        if reg_cost_match:
            try:
                data['registration_cost'] = float(reg_cost_match.group(1).replace(',', ''))
            except ValueError:
                pass

        return {f"subnet_{subnet_id}": data} if data else {}

    def _extract_homepage(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract data from taostats.io homepage.

        Homepage shows:
        - TAO price
        - Network statistics (block number, total stake, etc.)
        - Top subnets overview
        """
        data = {}

        # Extract TAO price (primary data point for taostats_price template)
        tao_patterns = [
            r'(?:TAO|Bittensor)\s*(?:Price)?[:\s]*\$?([\\d,\\.]+)',
            r'\$\s*([\d,\\.]+)\s*(?:USD)?(?:\s*per TAO|\s*TAO|\s*/TAO)?',
            r'Price[:\s]*\$?([\d,\\.]+)',
        ]
        for pattern in tao_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1).replace(',', ''))
                    # TAO price sanity check: should be between $10 and $10000
                    if 10 < price < 10000:
                        data['tao_price'] = price
                        data['current_price'] = price
                        break
                except ValueError:
                    continue

        # Extract 24h change for TAO
        change_patterns = [
            r'([+\-]?\d+\.?\d*)\s*%\s*(?:24h|day|daily)',
            r'(?:24h|Change)[:\s]*([+\-]?\d+\.?\d*)\s*%',
        ]
        for pattern in change_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    data['change_24h'] = float(match.group(1))
                    break
                except ValueError:
                    continue

        # Extract current block number
        # Patterns: "Block: 7,415,223" or "Current Block 7415223" or just large numbers
        block_patterns = [
            r'(?:Block|Current\s*Block|Block\s*Height|Chain\s*Height|Finalized)[:\s#]*([\d,]+)',
            r'#\s*([\d,]+)\s*(?:block|blocks)?',
            r'(?:Block|Height)[:\s]*([\d,]+)',
        ]
        for pattern in block_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    block = int(match.group(1).replace(',', ''))
                    # Sanity check: Bittensor blocks should be in millions
                    if block > 1000000:
                        data['current_block'] = block
                        break
                except ValueError:
                    continue

        # Fallback: Look for any 7+ digit number (block numbers are typically 7+ digits)
        if 'current_block' not in data:
            large_numbers = re.findall(r'\b(\d{7,})\b', content)
            for num_str in large_numbers:
                try:
                    num = int(num_str)
                    # Bittensor block range check (should be 1M - 100M roughly)
                    if 1000000 < num < 100000000:
                        data['current_block'] = num
                        break
                except ValueError:
                    continue

        # Extract subnet count from homepage
        # Patterns: "64 Subnets" or "Subnets: 64"
        subnet_patterns = [
            r'(\d+)\s*(?:Subnets|Active\s*Subnets)',
            r'(?:Subnets|Total\s*Subnets)[:\s]*(\d+)',
            r'(?:Subnet|Subnets)[:\s]*(\d+)',
        ]
        for pattern in subnet_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    count = int(match.group(1))
                    if 0 < count < 1000:  # Sanity check
                        data['subnet_count'] = count
                        break
                except ValueError:
                    continue

        # Fallback: count subnet mentions
        if 'subnet_count' not in data:
            subnet_refs = re.findall(r'(?:Subnet\s*#?\s*(\d+)|SN\s*(\d+))', content, re.IGNORECASE)
            if subnet_refs:
                # Get unique subnet IDs
                unique_ids = set()
                for groups in subnet_refs:
                    for g in groups:
                        if g:
                            try:
                                sid = int(g)
                                if 0 < sid < 100:  # Valid subnet range
                                    unique_ids.add(sid)
                            except ValueError:
                                pass
                if len(unique_ids) > 5:  # Only if we found multiple
                    data['subnet_count'] = len(unique_ids)

        # Extract network statistics
        total_stake_match = re.search(r'(?:Total Stake|Stake)[:\s]*([\d,\\.]+)\s*(?:τ|TAO)?', content, re.IGNORECASE)
        if total_stake_match:
            try:
                data['total_stake'] = float(total_stake_match.group(1).replace(',', ''))
            except ValueError:
                pass

        return {"taostats": data} if data else {}

    def _extract_subnet_list(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract data from subnet list page.

        Shows table of all subnets with basic info.
        Also extracts total subnet count.
        """
        result = {}

        # Count total subnets from the table
        # Look for subnet entries in the table
        subnet_entries = re.findall(r'\bSubnet\s*#?\s*(\d+)', content, re.IGNORECASE)
        if subnet_entries:
            # Get unique subnet IDs
            unique_ids = set(int(s) for s in subnet_entries if int(s) > 0)
            if unique_ids:
                result['taostats'] = {'subnet_count': len(unique_ids)}

        # Also try to count from table rows
        # Pattern: "27 | Compute | 0.5% | ..."
        subnet_pattern = re.compile(
            r'\b(\d{1,3})\b\s*[\|\-]\s*([A-Za-z0-9\s\-_]+?)\s*[\|\-]',
            re.IGNORECASE
        )

        subnet_count = 0
        for match in subnet_pattern.finditer(content):
            try:
                subnet_id = match.group(1)
                name = match.group(2).strip()
                if name and len(name) > 1 and int(subnet_id) > 0:
                    result[f"subnet_{subnet_id}"] = {
                        'name': name,
                        'subnet_name': name,
                    }
                    subnet_count += 1
            except (ValueError, IndexError):
                continue

        # Update subnet count if we found more from table parsing
        if subnet_count > 0:
            if 'taostats' not in result:
                result['taostats'] = {}
            result['taostats']['subnet_count'] = max(
                result.get('taostats', {}).get('subnet_count', 0),
                subnet_count
            )

        return result
