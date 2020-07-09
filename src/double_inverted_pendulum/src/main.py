#!/usr/bin/env python3
from cart_controller import cart_controller

cart_controller = cart_controller("controller_commander", 100)

while True:
    cart_controller.actuate_wheels(1000)