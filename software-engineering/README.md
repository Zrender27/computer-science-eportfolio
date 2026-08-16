# Smart Thermostat Controller
<img width="1536" height="1024" alt="ChatGPT Image Aug 16, 2026 at 03_43_10 PM" src="https://github.com/user-attachments/assets/c8696871-4eb2-4be2-a4f5-559d820bb44a" />

*AI-generated visual created with ChatGPT to represent the Smart Thermostat software engineering enhancement.*
## Software Design and Engineering Artifact

This project is a Raspberry Pi-based smart thermostat controller developed in Python. The application integrates software with physical hardware to monitor temperature, manage thermostat states, accept user input, display system information, and communicate thermostat data through a serial connection.

This repository includes both the **original version** of the application and an **enhanced version** demonstrating improvements in software design, maintainability, reliability, and defensive programming.

## Project Features

The thermostat application uses:

* GPIO buttons for user input
* AHTx0 temperature sensor for environmental readings
* PWM-controlled LEDs to represent heating and cooling states
* I2C character LCD for thermostat information
* UART serial communication for transmitting timestamped thermostat data
* Python-based thermostat control logic

## Original Artifact

The original application demonstrates a functional embedded-system workflow. However, much of the thermostat state is managed through global variables and standalone functions.

This design works for a relatively small application but becomes more difficult to maintain as additional functionality is introduced.

**File:** `CS350_Original_Smart_Thermostat.py`

## Enhanced Artifact

The enhanced version preserves the purpose of the original thermostat while improving the internal software design.

Major enhancements include:

* Object-oriented `ThermostatController` architecture
* `ThermostatMode` enumeration for clearly defined operating states
* Reduced dependence on global variables
* Centralized configuration using named constants
* Temperature limits between 55°F and 85°F
* Validation of sensor readings
* Improved exception and communication-error handling
* Hardware resource cleanup when the application terminates
* Hysteresis control to reduce unnecessary state switching around the target temperature
* Clear code comments documenting significant enhancements

These changes make the application easier to understand, maintain, test, and extend.

## Design Considerations

One important enhancement was the introduction of a small hysteresis range around the thermostat set point. Without hysteresis, small temperature fluctuations near the target temperature could repeatedly switch the system between operating states.

The enhanced implementation uses a half-degree dead band to provide more stable behavior. This represents a design trade-off between immediate responsiveness and system stability.

The project was intentionally maintained as a relatively compact application rather than being divided into numerous modules. This keeps the embedded application manageable while still demonstrating object-oriented design and separation of responsibilities.

## Defensive Programming and Security

Although this is not primarily a security-focused application, the enhanced version applies several defensive programming practices.

These include:

* Validating external sensor data before using it
* Restricting user-controlled temperature values
* Handling serial communication failures
* Preventing invalid thermostat states
* Returning hardware resources to a safe state when the program exits

These practices reduce the likelihood that unexpected input or hardware communication problems will cause uncontrolled application behavior.

## Requirements

The project is designed to run on a Raspberry Pi with compatible hardware connected.

### Hardware

* Raspberry Pi
* AHTx0-compatible temperature sensor
* I2C character LCD
* GPIO push buttons
* LEDs
* Appropriate resistors and wiring
* UART-compatible serial connection as required by the original hardware configuration

### Software

* Python 3
* Raspberry Pi GPIO support
* CircuitPython-compatible AHTx0 library
* LCD library used by the project
* PySerial or compatible serial communication package

## Running the Project

1. Connect the required thermostat hardware to the Raspberry Pi using the GPIO, I2C, and UART configuration defined in the Python source file.
2. Install Python 3 and the required hardware libraries.
3. Download or clone this repository.
4. Navigate to the `software-engineering` directory.
5. Run the enhanced application using:

```bash
python3 CS350_Enhanced_Smart_Thermostat.py
```

The original version can be reviewed or executed using:

```bash
python3 CS350_Original_Smart_Thermostat.py
```

## Testing Note

The enhanced Python source has been checked for valid Python syntax. Because the application depends on physical Raspberry Pi components, complete GPIO, sensor, LCD, LED, and UART behavior requires testing with the corresponding hardware.

## Skills Demonstrated

This enhancement demonstrates experience with:

* Python
* Object-oriented programming
* Embedded software development
* Hardware/software integration
* State management
* Event-driven programming
* Input and sensor validation
* Exception handling
* Resource management
* Defensive programming
* Software maintainability
* Technical documentation

## Enhancement Narrative

A separate enhancement narrative accompanies this artifact and explains the development process, design decisions, challenges, course outcomes, and reflection associated with the enhanced application.

## Author

**Zoe Render**
Computer Science | Software Engineering
