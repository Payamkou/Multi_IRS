from config import Config, config_hash, seed_for


def test_defaults_match_spec():
    c = Config()
    assert c.M == 80 and c.K == 6
    assert c.crn_weights == (0.90, 0.80, 0.85)
    assert c.irs_pos == ((10., 10.), (10., -10.))
    assert c.sweep_crn_dbm == (20, 25, 30, 35, 40, 45)


def test_json_roundtrip(tmp_path):
    c = Config(M=40)
    p = tmp_path / "c.json"
    c.to_json(p)
    d = Config.from_json(p)
    assert d == c
    assert isinstance(d.irs_pos, tuple) and isinstance(d.irs_pos[0], tuple)
    assert isinstance(d.sweep_M, tuple)


def test_yaml_written(tmp_path):
    p = tmp_path / "c.yaml"
    Config().to_yaml(p)
    assert "M: 80" in p.read_text()


def test_config_hash_changes_with_field():
    a, b = Config(), Config(M=40)
    assert config_hash(a, ("M",)) != config_hash(b, ("M",))
    assert config_hash(a, ("K",)) == config_hash(b, ("K",))


def test_seed_registry():
    assert (seed_for("train"), seed_for("val"), seed_for("test")) == (1001, 2001, 3001)
    assert seed_for("sweep_M", 20) == 40020
    assert seed_for("sweep_K", 10) == 50010
    assert seed_for("sweep_power", 25) == 60025
    assert seed_for("model_init") == 9001
