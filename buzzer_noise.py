import RPi.GPIO as GPIO
import time

# Setup
GPIO.setmode(GPIO.BCM)
gpio_pin_for_pwm = 13
GPIO.setup(gpio_pin_for_pwm, GPIO.OUT)

# Creează PWM (frecvență în Hz)
pwm_signal = GPIO.PWM(gpio_pin_for_pwm, 1000)  # 1kHz

# Pornește PWM (duty cycle 50%)
pwm_signal.start(50)

try:
    # Buzzer activ timp de 5 secunde
    time.sleep(5)

    # Schimbă frecvența (pentru buzzer pasiv -> sunet diferit)
    pwm_signal.ChangeFrequency(2000)
    time.sleep(5)

finally:
    pwm_signal.stop()
    GPIO.cleanup()