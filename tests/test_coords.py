from stomclient.coords import widget_to_image


def test_top_left_anchored_unscaled():
    # widget 200x100, image 100x100 -> scale=min(2,1)=1, drawn top-left 100x100
    assert widget_to_image((50, 50), (200, 100), (100, 100)) == (50.0, 50.0)


def test_scaled_down():
    # widget 200x200, image 100x100 -> scale=2 -> click (100,100) maps to (50,50)
    assert widget_to_image((100, 100), (200, 200), (100, 100)) == (50.0, 50.0)


def test_outside_image_returns_none():
    assert widget_to_image((150, 50), (200, 100), (100, 100)) is None
