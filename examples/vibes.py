# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic==0.51.0",
#     "jinja2==3.1.6",
#     "mohtml==0.1.10",
# ]
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from mohtml import div
    import datetime as dt
    return div, dt


@app.cell
def _(div, dt, render_weather_info):
    class Weather:
        def __init__(self, temperature, humidity, rainfall):
            self.temperature = temperature
            self.humidity = humidity
            self.rainfall = rainfall

        def _display_(self):
            return div(render_weather_info(temperature=self.temperature, humidity=self.humidity, rainfall=self.rainfall, now=dt.datetime.now()))
    return (Weather,)


@app.cell
def _(Weather):
    Weather(temperature=25, humidity=60, rainfall=0)
    return


@app.cell
def _():
    import jinja2

    def create_weather_template():
        """
        Creates a Jinja2 template for rendering weather information with SVG icons.
    
        Returns:
            jinja2.Template: A template that can render weather information
        """
        template_str = """
        <div style="font-family: Arial, sans-serif; padding: 20px; border-radius: 10px; background-color: #f0f8ff; max-width: 400px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="color: #2c3e50; text-align: center;">Weather Information</h2>
        
            <div style="display: flex; align-items: center; margin: 15px 0;">
                <div style="margin-right: 15px;">
                    <!-- Temperature SVG Icon -->
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#e74c3c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"></path>
                        <circle cx="11.5" cy="18" r="2.5"></circle>
                    </svg>
                </div>
                <div>
                    <span style="font-size: 18px; font-weight: bold;">Temperature:</span>
                    <span style="font-size: 18px; margin-left: 5px;">{{ temperature }}°C</span>
                </div>
            </div>
        
            <div style="display: flex; align-items: center; margin: 15px 0;">
                <div style="margin-right: 15px;">
                    <!-- Humidity SVG Icon -->
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3498db" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"></path>
                    </svg>
                </div>
                <div>
                    <span style="font-size: 18px; font-weight: bold;">Humidity:</span>
                    <span style="font-size: 18px; margin-left: 5px;">{{ humidity }}%</span>
                </div>
            </div>
        
            <div style="display: flex; align-items: center; margin: 15px 0;">
                <div style="margin-right: 15px;">
                    <!-- Rainfall SVG Icon -->
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#2980b9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="16" y1="13" x2="16" y2="21"></line>
                        <line x1="8" y1="13" x2="8" y2="21"></line>
                        <line x1="12" y1="15" x2="12" y2="23"></line>
                        <path d="M20 16.58A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25"></path>
                    </svg>
                </div>
                <div>
                    <span style="font-size: 18px; font-weight: bold;">Rainfall:</span>
                    <span style="font-size: 18px; margin-left: 5px;">{{ rainfall }} mm</span>
                </div>
            </div>
        
            <div style="text-align: center; margin-top: 20px; font-size: 14px; color: #7f8c8d;">
                Last updated: {{ now.strftime('%Y-%m-%d %H:%M:%S') if now else 'N/A' }}
            </div>
        </div>
        """
        return jinja2.Template(template_str)

    def render_weather_info(temperature, humidity, rainfall, now=None):
        """
        Renders weather information using the template.
    
        Args:
            temperature (float): Temperature in Celsius
            humidity (float): Humidity percentage
            rainfall (float): Rainfall in millimeters
            now (datetime, optional): Current datetime. Defaults to None.
        
        Returns:
            str: HTML representation of the weather information
        """
        template = create_weather_template()
        return template.render(
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            now=now
        )
    return (render_weather_info,)


if __name__ == "__main__":
    app.run()
