#!/bin/bash

coverage run -m unittest discover tests/unit_test/
coverage report -m