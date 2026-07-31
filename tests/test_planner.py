from harmobridge.planner import Operation, compile_segments


def test_planner_places_nonlinearity_in_share_domain():
    operations = [
        Operation("affine", (1,)),
        Operation("relu", (1,)),
        Operation("outer_product", (1, 1)),
        Operation("trunc", (2,)),
    ]
    segments = compile_segments(operations, degree_budget=2)
    assert [segment.domain for segment in segments] == ["harmonic", "share", "harmonic", "share"]
