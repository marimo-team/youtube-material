import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    from mofresh import refresh_altair, HTMLRefreshWidget, altair2svg


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import datetime as dt

    import subprocess 
    import time
    return dt, mo, pl, subprocess, time


@app.cell
def _():
    from enviroplus import gas
    return (gas,)


@app.cell
def _(dt, subprocess):
    def read_temp():
        resp = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True)
        temp = float(resp.stdout.decode().replace("temp=", "")[:4])
        return {"temp": temp, "time": dt.datetime.now()}
    return


@app.cell
def _(mo, sample):
    get_state, set_state = mo.state([sample()])
    return get_state, set_state


@app.cell
def _(get_state, sample, set_state):
    def read_moar():
        state = get_state()
        state.append(sample())
        set_state(state[-200:])
    return (read_moar,)


@app.cell
def _(dt, gas, subprocess):
    from bme280 import BME280
    from smbus2 import SMBus
    from ltr559 import LTR559

    bus = SMBus(1)
    bme280 = BME280(i2c_dev=bus)
    ltr559 = LTR559()

    def sample():
        _ = gas.read_all()
        _.nh3, _.oxidising, _.reducing

        temperature = bme280.get_temperature()
        pressure = bme280.get_pressure()
        humidity = bme280.get_humidity()
        resp = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True)
        cpu_temp = float(resp.stdout.decode().replace("temp=", "")[:4])

        return {
            "nh3": _.nh3, 
            "oxidising": _.oxidising,
            "reducing": _.reducing, 
            "temp_sensor": temperature, 
            "pressure": pressure, 
            "humidity": humidity, 
            "temp_cpu": cpu_temp, 
            "lux": ltr559.get_lux(), 
            "proximity": ltr559.get_proximity(),
            "time": dt.datetime.now()
        }
    return (sample,)


@app.cell
def _(sample):
    sample()
    return


@app.cell
def _(HTMLRefreshWidget, alt, pl, refresh_altair, sample):
    @refresh_altair
    def altair_chart(data):
        df = pl.DataFrame(data)
        p1 = alt.Chart(df).mark_line().encode(x="time", y="temp_cpu")
        p2 = alt.Chart(df).mark_line().encode(x="time", y="temp_sensor")
        p3 = alt.Chart(df).mark_line().encode(x="time", y="lux")
        return (p1 + p2) | p3

    svg_widget = HTMLRefreshWidget(html=altair_chart([sample()]))
    svg_widget
    return altair_chart, svg_widget


@app.cell
def _(get_state):
    get_state()[-1]
    return


@app.cell
def _(altair_chart, get_state, read_moar, svg_widget, time):
    for i in range(10):
        time.sleep(0.5)
        read_moar()
        svg_widget.html = altair_chart(get_state())
    return


if __name__ == "__main__":
    app.run()
