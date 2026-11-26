

def test_barrel_symbol_compiles_to_an_explosive_barrel():
    from training.gen.compile import compile_grid

    text = compile_grid("#######\n#P.x.E#\n#######")
    assert "type = 2035;" in text
