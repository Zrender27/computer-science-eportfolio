from gpiozero import Button, PWMLED
from time import sleep, strftime
import serial
import board
import adafruit_ahtx0
from RPLCD.i2c import CharLCD

# GPIO setup
red_led = PWMLED(17)
blue_led = PWMLED(27)

mode_button = Button(22)
increase_button = Button(23)
decrease_button = Button(24)

# I2C temperature sensor
i2c = board.I2C()
sensor = adafruit_ahtx0.AHTx0(i2c)

# LCD setup - adjust address if needed
lcd = CharLCD('PCF8574', 0x27)

# UART setup
uart = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)

# Thermostat variables
states = ["OFF", "HEATING", "COOLING"]
state_index = 0
set_temp = 72


def change_mode():
    global state_index
    state_index = (state_index + 1) % len(states)


def increase_temp():
    global set_temp
    set_temp += 1


def decrease_temp():
    global set_temp
    set_temp -= 1


mode_button.when_pressed = change_mode
increase_button.when_pressed = increase_temp
decrease_button.when_pressed = decrease_temp


def fahrenheit(celsius):
    return (celsius * 9 / 5) + 32


def update_leds(mode, current_temp):
    red_led.off()
    blue_led.off()

    if mode == "HEATING":
        if current_temp < set_temp:
            red_led.pulse(fade_in_time=1, fade_out_time=1)
        else:
            red_led.value = 1

    elif mode == "COOLING":
        if current_temp > set_temp:
            blue_led.pulse(fade_in_time=1, fade_out_time=1)
        else:
            blue_led.value = 1


def update_lcd(mode, current_temp):
    lcd.clear()
    lcd.write_string(strftime("%m/%d %H:%M"))
    lcd.crlf()
    lcd.write_string(f"{mode} {current_temp:.1f}F/{set_temp}F")


def send_uart(mode, current_temp):
    message = f"{strftime('%Y-%m-%d %H:%M:%S')}, Mode:{mode}, Temp:{current_temp:.1f}F, Set:{set_temp}F\n"
    uart.write(message.encode("utf-8"))
    print(message)


try:
    while True:
        current_temp = fahrenheit(sensor.temperature)
        current_mode = states[state_index]

        update_leds(current_mode, current_temp)
        update_lcd(current_mode, current_temp)
        send_uart(current_mode, current_temp)

        sleep(2)

except KeyboardInterrupt:
    red_led.off()
    blue_led.off()
    lcd.clear()
    uart.close()