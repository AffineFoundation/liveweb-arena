"""Weather page content extractor for GT extraction"""

import re
from typing import Any, Dict
from urllib.parse import urlparse, unquote

from liveweb_arena.core.gt_extraction import PageExtractor


class WeatherExtractor(PageExtractor):
    """Extract weather data from wttr.in pages"""

    @property
    def source_name(self) -> str:
        return "weather"

    def matches_url(self, url: str) -> bool:
        url_lower = url.lower()
        return "wttr.in" in url_lower

    def classify_page(self, url: str) -> str:
        """Classify page type.

        wttr.in pages:
        - https://wttr.in/ (homepage, shows user's location)
        - https://wttr.in/London (detail page for specific location)
        - https://v2.wttr.in/Tokyo (v2 format detail page)
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return "homepage"
        # Location-specific pages are "detail" pages
        return "detail"

    def extract(self, url: str, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract weather data from wttr.in page accessibility tree."""
        page_type = self.classify_page(url)

        if page_type == "detail":
            return self._extract_detail(url, content)
        elif page_type == "homepage":
            return self._extract_homepage(content)
        return {}

    def _extract_location(self, url: str) -> str:
        """Extract location from URL path."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        if not path:
            return ""

        # Decode URL encoding
        location = unquote(path)

        # Remove format suffixes if present
        location = re.sub(r'\?.*$', '', location)

        # Normalize: lowercase, replace + with space
        location = location.replace('+', ' ').lower()

        return location

    def _extract_detail(self, url: str, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract data from location detail page.

        wttr.in shows weather data in text format including:
        - Current temperature
        - Feels like temperature
        - Humidity
        - Wind speed
        - Chance of rain
        - Weather condition descriptions

        The accessibility tree typically contains lines like:
        - "Temperature: 22°C"
        - "Feels Like: 24°C"
        - "Humidity: 65%"
        - "Wind: 15 km/h"
        """
        location = self._extract_location(url)
        if not location:
            # Try to extract location from content
            loc_match = re.search(r'Weather report:\s*(\S+)', content, re.IGNORECASE)
            if loc_match:
                location = loc_match.group(1).lower()
            else:
                location = "unknown"

        data = {}

        # Extract current temperature
        # wttr.in shows temperature in various formats depending on output mode
        # Common patterns: "22°C", "+22°C", "Temperature: 22", "22 °C"
        temp_patterns = [
            r'(?:Temperature|Temp)[:\s]*([+\-]?\d+)\s*°?C',
            r'([+\-]?\d+)\s*°\s*C\b',  # "22 °C" or "22°C"
            r'([+\-]?\d+)°C',  # Compact format
            r'\b([+\-]?\d{1,2})\s*(?:degrees|deg)\s*(?:C|Celsius)?',
            # wttr.in specific: look for temperature in the main display
            r'(?:^|\n)\s*([+\-]?\d{1,2})\s*$',  # Standalone number on a line
        ]
        for pattern in temp_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for m in matches:
                try:
                    temp = int(m)
                    # Sanity check: reasonable temperature range (-50 to +60°C)
                    if -50 <= temp <= 60:
                        data['temperature'] = temp
                        data['tempC'] = temp
                        break
                except ValueError:
                    continue
            if 'temperature' in data:
                break

        # Extract feels like temperature
        feels_match = re.search(r'(?:Feels\s*Like|Feels)[:\s]*([+\-]?\d+)\s*°?C?', content, re.IGNORECASE)
        if feels_match:
            try:
                data['feels_like'] = int(feels_match.group(1))
                data['FeelsLikeC'] = data['feels_like']
            except ValueError:
                pass

        # Extract humidity
        humidity_patterns = [
            r'(?:Humidity)[:\s]*(\d+)\s*%?',
            r'(\d+)\s*%\s*(?:humidity|hum)',
            r'(\d{1,3})\s*%',  # Any percentage (pick reasonable one)
        ]
        for pattern in humidity_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for m in matches:
                try:
                    hum = int(m)
                    # Sanity check: humidity 0-100%
                    if 0 <= hum <= 100:
                        data['humidity'] = hum
                        break
                except ValueError:
                    continue
            if 'humidity' in data:
                break

        # Extract wind speed
        wind_patterns = [
            r'(?:Wind)[:\s]*(\d+)\s*(?:km/?h|kmh|kph)',
            r'(?:Wind speed)[:\s]*(\d+)',
            r'(\d+)\s*km/?h\s*(?:wind)?',
        ]
        for pattern in wind_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    data['wind_speed'] = int(match.group(1))
                    data['windspeedKmph'] = data['wind_speed']
                    break
                except ValueError:
                    continue

        # Extract chance of rain / precipitation
        rain_match = re.search(r'(?:Rain|Precipitation|Chance of rain)[:\s]*(\d+)\s*%?', content, re.IGNORECASE)
        if rain_match:
            try:
                data['precipitation_chance'] = int(rain_match.group(1))
                data['chanceofrain'] = data['precipitation_chance']
            except ValueError:
                pass

        # Extract high/max temperature
        high_patterns = [
            r'(?:High|Max|Maximum)[:\s]*([+\-]?\d+)\s*°?C?',
            r'(?:High|Max)[:\s]*([+\-]?\d+)',
        ]
        for pattern in high_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    data['temperature_high'] = int(match.group(1))
                    data['maxtempC'] = data['temperature_high']
                    break
                except ValueError:
                    continue

        # Extract low/min temperature
        low_patterns = [
            r'(?:Low|Min|Minimum)[:\s]*([+\-]?\d+)\s*°?C?',
            r'(?:Low|Min)[:\s]*([+\-]?\d+)',
        ]
        for pattern in low_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    data['temperature_low'] = int(match.group(1))
                    data['mintempC'] = data['temperature_low']
                    break
                except ValueError:
                    continue

        # Extract UV index
        uv_match = re.search(r'(?:UV|UV Index)[:\s]*(\d+)', content, re.IGNORECASE)
        if uv_match:
            try:
                data['uv_index'] = int(uv_match.group(1))
                data['uvIndex'] = data['uv_index']
            except ValueError:
                pass

        # Extract visibility
        vis_match = re.search(r'(?:Visibility)[:\s]*(\d+)\s*(?:km)?', content, re.IGNORECASE)
        if vis_match:
            try:
                data['visibility'] = int(vis_match.group(1))
            except ValueError:
                pass

        # Extract cloud cover
        cloud_match = re.search(r'(?:Cloud|Clouds|Cloud cover)[:\s]*(\d+)\s*%?', content, re.IGNORECASE)
        if cloud_match:
            try:
                data['cloud_cover'] = int(cloud_match.group(1))
                data['cloudcover'] = data['cloud_cover']
            except ValueError:
                pass

        # Extract pressure
        pressure_match = re.search(r'(?:Pressure)[:\s]*(\d+)\s*(?:hPa|mb)?', content, re.IGNORECASE)
        if pressure_match:
            try:
                data['pressure'] = int(pressure_match.group(1))
            except ValueError:
                pass

        # Extract weather condition description
        condition_patterns = [
            r'(?:Condition|Weather)[:\s]*(Sunny|Cloudy|Rainy|Clear|Overcast|Partly cloudy|Light rain|Heavy rain|Snow|Fog|Mist|Thunderstorm)',
        ]
        for pattern in condition_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                data['condition'] = match.group(1).strip()
                data['weatherDesc'] = data['condition']
                break

        return {location: data} if data else {}

    def _extract_homepage(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract data from wttr.in homepage.

        Homepage shows weather for detected location.
        Less reliable than specific location pages.
        """
        # Try to find the location name from content
        loc_match = re.search(r'Weather report:\s*(\S+)', content, re.IGNORECASE)
        if loc_match:
            location = loc_match.group(1).lower()
            # Use the same extraction logic as detail pages
            return self._extract_detail(f"https://wttr.in/{location}", content)

        return {}
