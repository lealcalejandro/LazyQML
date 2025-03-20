#!/usr/bin/env python

"""Tests for `lazyqml` package."""

import unittest

class TestLazyqml(unittest.TestCase):
    """Tests for `lazyqml` package."""

    def test_import(self):
        import lazyqml 
        # print("Imported correctly")

    def test_simulation_strings(self):
        # Verify getter/setter of simulation type flag
        from lazyqml import lazyqml

        sim = "statevector"
        lazyqml.set_simulation_type(sim)
        self.assertTrue(lazyqml.get_simulation_type(), "statevector")

        sim = "tensor"
        lazyqml.set_simulation_type(sim)
        self.assertTrue(lazyqml.get_simulation_type(), "tensor")

        # Verify that ValueError is raised when number or different string is set
        sim = 3
        with self.assertRaises(ValueError):
            lazyqml.set_simulation_type(sim)

        sim = "tns"
        with self.assertRaises(ValueError):
            lazyqml.set_simulation_type(sim)

if __name__ == '__main__':
    unittest.main()