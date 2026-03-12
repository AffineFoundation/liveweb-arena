from liveweb_arena.core.agent_policy import AgentPolicy


def test_parse_response_recovers_missing_trailing_brace():
    policy = AgentPolicy()
    raw = (
        '{"action":{"type":"stop","params":{"format":"json","final":{"answers":{"answer1":"4194"}}}}'
    )

    action = policy.parse_response(raw)

    assert action is not None
    assert action.action_type == "stop"
    assert action.params["format"] == "json"
    assert action.params["final"]["answers"]["answer1"] == "4194"


def test_parse_response_still_rejects_non_json_text():
    policy = AgentPolicy()
    raw = "Please click search and continue."

    action = policy.parse_response(raw)
    assert action is None
