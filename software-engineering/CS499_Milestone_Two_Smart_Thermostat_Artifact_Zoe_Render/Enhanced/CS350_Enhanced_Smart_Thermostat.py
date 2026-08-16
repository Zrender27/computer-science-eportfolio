"""
Zoe Render
CS 499 Milestone Two - Software Design and Engineering Enhancement
Enhanced Raspberry Pi Thermostat Controller

This version preserves the behavior of the original thermostat artifact while
improving maintainability, safety, reliability, and separation of concerns.

NOTE: This program is intended to run on a Raspberry Pi with the same hardware
used by the original project (GPIO buttons/LEDs, AHTx0 sensor, I2C LCD, UART).
"""

from gpiozero import Button, PWMLED
from time import sleep, strftime
from enum import Enum
import math
import serial
import board
import adafruit_ahtx0
from RPLCD.i2c import CharLCD


# ============================================================================
# ENHANCEMENT 1: CENTRALIZED CONFIGURATION
# Hardware pins, serial settings, and thermostat limits are now constants.
# This removes "magic numbers" from the program and makes maintenance easier.
# ============================================================================
RED_LED_PIN = 17
BLUE_LED_PIN = 27
MODE_BUTTON_PIN = 22
INCREASE_BUTTON_PIN = 23
DECREASE_BUTTON_PIN = 24

LCD_ADDRESS = 0x27
UART_PORT = "/dev/serial0"
UART_BAUD_RATE = 9600

DEFAULT_SET_TEMP = 72
MIN_SET_TEMP = 55
MAX_SET_TEMP = 85
TEMPERATURE_STEP = 1
HYSTERESIS = 0.5
LOOP_DELAY_SECONDS = 2


# ============================================================================
# ENHANCEMENT 2: ENUM FOR THERMOSTAT MODES
# The original project stored mode names as strings in a list. An Enum prevents
# accidental invalid mode values and makes the state logic more explicit.
# ============================================================================
class ThermostatMode(Enum):
    OFF = "OFF"
    HEATING = "HEATING"
    COOLING = "COOLING"


# ============================================================================
# ENHANCEMENT 3: OBJECT-ORIENTED CONTROLLER DESIGN
# The original program relied on global state_index and set_temp variables.
# Encapsulating thermostat state and behavior in one class reduces global state,
# improves cohesion, and makes future testing and extension easier.
# ============================================================================
class ThermostatController:
    """Controls thermostat state, hardware output, display, and telemetry."""

    def __init__(self):
        self.modes = list(ThermostatMode)
        self.mode_index = 0
        self.set_temp = DEFAULT_SET_TEMP

        # Hardware initialization is grouped together so dependencies are clear.
        self.red_led = PWMLED(RED_LED_PIN)
        self.blue_led = PWMLED(BLUE_LED_PIN)

        self.mode_button = Button(MODE_BUTTON_PIN, bounce_time=0.1)
        self.increase_button = Button(INCREASE_BUTTON_PIN, bounce_time=0.1)
        self.decrease_button = Button(DECREASE_BUTTON_PIN, bounce_time=0.1)

        self.i2c = board.I2C()
        self.sensor = adafruit_ahtx0.AHTx0(self.i2c)
        self.lcd = CharLCD("PCF8574", LCD_ADDRESS)
        self.uart = serial.Serial(
            UART_PORT,
            baudrate=UART_BAUD_RATE,
            timeout=1,
        )

        # Button callbacks remain event-driven, as in the original artifact.
        self.mode_button.when_pressed = self.change_mode
        self.increase_button.when_pressed = self.increase_temp
        self.decrease_button.when_pressed = self.decrease_temp

    @property
    def current_mode(self):
        """Return the currently selected thermostat mode."""
        return self.modes[self.mode_index]

    def change_mode(self):
        """Cycle through OFF, HEATING, and COOLING modes."""
        self.mode_index = (self.mode_index + 1) % len(self.modes)

    # ========================================================================
    # ENHANCEMENT 4: INPUT VALIDATION / SAFE TEMPERATURE BOUNDS
    # The original increase/decrease functions could raise or lower the set
    # point forever. The enhanced version constrains the user-selected target
    # to a defined safe operating range.
    # ========================================================================
    def increase_temp(self):
        self.set_temp = min(
            self.set_temp + TEMPERATURE_STEP,
            MAX_SET_TEMP,
        )

    def decrease_temp(self):
        self.set_temp = max(
            self.set_temp - TEMPERATURE_STEP,
            MIN_SET_TEMP,
        )

    @staticmethod
    def fahrenheit(celsius):
        """Convert Celsius to Fahrenheit."""
        return (celsius * 9 / 5) + 32

    # ========================================================================
    # ENHANCEMENT 5: SENSOR VALIDATION AND EXCEPTION HANDLING
    # Sensor data is checked before it reaches thermostat logic. Invalid or
    # non-finite readings raise a controlled error instead of silently driving
    # heating/cooling behavior with unreliable data.
    # ========================================================================
    def read_temperature(self):
        celsius = self.sensor.temperature

        if celsius is None or not math.isfinite(celsius):
            raise ValueError("Temperature sensor returned an invalid reading.")

        fahrenheit_value = self.fahrenheit(celsius)

        # A broad plausibility check catches clearly corrupted readings while
        # still allowing realistic environmental temperatures.
        if fahrenheit_value < -40 or fahrenheit_value > 185:
            raise ValueError(
                f"Temperature reading out of expected range: "
                f"{fahrenheit_value:.1f}F"
            )

        return fahrenheit_value

    # ========================================================================
    # ENHANCEMENT 6: HYSTERESIS TO REDUCE RAPID OUTPUT SWITCHING
    # A small dead-band around the set point prevents the system from rapidly
    # toggling as sensor readings fluctuate around the target temperature.
    # ========================================================================
    def update_leds(self, mode, current_temp):
        self.red_led.off()
        self.blue_led.off()

        if mode == ThermostatMode.HEATING:
            if current_temp < self.set_temp - HYSTERESIS:
                self.red_led.pulse(fade_in_time=1, fade_out_time=1)
            else:
                self.red_led.value = 1

        elif mode == ThermostatMode.COOLING:
            if current_temp > self.set_temp + HYSTERESIS:
                self.blue_led.pulse(fade_in_time=1, fade_out_time=1)
            else:
                self.blue_led.value = 1

    # ========================================================================
    # ENHANCEMENT 7: DISPLAY FORMATTING IS ISOLATED IN ITS OWN METHOD
    # Keeping presentation logic separate from sensor/control logic makes the
    # program easier to modify for a different display in the future.
    # ========================================================================
    def update_lcd(self, mode, current_temp):
        self.lcd.clear()
        self.lcd.write_string(strftime("%m/%d %H:%M"))
        self.lcd.crlf()

        # Keep the second line concise for a character LCD.
        display_text = (
            f"{mode.value} {current_temp:.1f}F/"
            f"{self.set_temp}F"
        )
        self.lcd.write_string(display_text[:16])

    # ========================================================================
    # ENHANCEMENT 8: STRUCTURED TELEMETRY WITH UART ERROR ISOLATION
    # The message retains the original timestamp/mode/temp/set-point data, but
    # serial transmission errors are handled so a UART issue does not crash the
    # entire thermostat control loop.
    # ========================================================================
    def send_uart(self, mode, current_temp):
        message = (
            f"{strftime('%Y-%m-%d %H:%M:%S')},"
            f" Mode:{mode.value},"
            f" Temp:{current_temp:.1f}F,"
            f" Set:{self.set_temp}F\n"
        )

        try:
            self.uart.write(message.encode("utf-8"))
        except serial.SerialException as error:
            print(f"UART error: {error}")

        print(message, end="")

    def show_sensor_error(self, error):
        """Provide a visible error message without terminating immediately."""
        print(f"Sensor error: {error}")

        try:
            self.lcd.clear()
            self.lcd.write_string("Sensor Error")
            self.lcd.crlf()
            self.lcd.write_string("Check hardware")
        except Exception as lcd_error:
            print(f"LCD error while reporting sensor problem: {lcd_error}")

    # ========================================================================
    # ENHANCEMENT 9: MAIN LOOP MOVED INTO THE CONTROLLER
    # This creates a clear application entry point and separates setup from the
    # continuous runtime behavior.
    # ========================================================================
    def run(self):
        while True:
            try:
                current_temp = self.read_temperature()
                mode = self.current_mode

                self.update_leds(mode, current_temp)
                self.update_lcd(mode, current_temp)
                self.send_uart(mode, current_temp)

            except (ValueError, OSError, RuntimeError) as error:
                # A bad sensor read no longer causes uncontrolled program exit.
                self.red_led.off()
                self.blue_led.off()
                self.show_sensor_error(error)

            sleep(LOOP_DELAY_SECONDS)

    # ========================================================================
    # ENHANCEMENT 10: RELIABLE RESOURCE CLEANUP
    # The original cleanup occurred only inside KeyboardInterrupt handling.
    # A dedicated cleanup method combined with finally ensures hardware is left
    # in a safe state even if another runtime exception occurs.
    # ========================================================================
    def cleanup(self):
        self.red_led.off()
        self.blue_led.off()

        try:
            self.lcd.clear()
        except Exception:
            pass

        if self.uart.is_open:
            self.uart.close()

        self.mode_button.close()
        self.increase_button.close()
        self.decrease_button.close()
        self.red_led.close()
        self.blue_led.close()


# ============================================================================
# ENHANCEMENT 11: EXPLICIT APPLICATION ENTRY POINT
# Using main() makes program startup obvious and improves readability/testability.
# ============================================================================
def main():
    thermostat = None

    try:
        thermostat = ThermostatController()
        thermostat.run()
    except KeyboardInterrupt:
        print("\nThermostat stopped by user.")
    except Exception as error:
        # Unexpected errors are reported rather than failing silently.
        print(f"Unexpected thermostat error: {error}")
    finally:
        if thermostat is not None:
            thermostat.cleanup()


if __name__ == "__main__":
    main()
