from liveweb_arena.plugins.coingecko.coingecko import CoinGeckoPlugin


def test_coingecko_aliases_normalize_common_coin_slugs():
    plugin = CoinGeckoPlugin()
    assert plugin._extract_coin_id("https://www.coingecko.com/en/coins/bnb") == "binancecoin"
    assert plugin._extract_coin_id("https://www.coingecko.com/en/coins/binance-coin") == "binancecoin"
    assert plugin._extract_coin_id("https://www.coingecko.com/en/coins/xrp") == "ripple"
    assert plugin._extract_coin_id("https://www.coingecko.com/en/coins/ada") == "cardano"
    assert plugin._extract_coin_id("https://www.coingecko.com/en/coins/polkadot-new") == "polkadot"
    assert plugin._extract_coin_id("https://www.coingecko.com/en/coins/near-protocol") == "near"
    assert plugin._extract_coin_id("https://www.coingecko.com/en/coins/hedera") == "hedera-hashgraph"
