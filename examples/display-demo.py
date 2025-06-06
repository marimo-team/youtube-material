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
    return (div,)


@app.cell
def _(div, render_weather_template):
    class Weather():
        def __init__(self, temperature, humidity, precipitation):
            self.temperature = temperature
            self.humidity = humidity
            self.precipitation = precipitation

        def _display_(self):
            return div(render_weather_template(self.temperature, self.humidity, self.precipitation))
    return (Weather,)


@app.cell
def _(Weather):
    Weather(temperature=15, humidity=100, precipitation=10.1)
    return


@app.cell
def _():
    import jinja2

    def render_weather_template(temperature, humidity, precipitation):
        """
        Render a weather information card using Jinja2 template.
    
        Args:
            temperature (float): Temperature in Celsius
            humidity (float): Humidity percentage (0-100)
            precipitation (float): Precipitation in mm
    
        Returns:
            str: Rendered HTML content
        """
        # Define the template
        template_str = """
            <style>
                .weather-card {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 400px;
                    margin: 20px auto;
                    padding: 20px;
                    border-radius: 15px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    background: linear-gradient(to bottom right, #f7f9fc, #e3eafd);
                    color: #333;
                }
                .weather-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }
                .temperature {
                    font-size: 3rem;
                    font-weight: bold;
                    margin: 0;
                }
                .weather-details {
                    display: flex;
                    justify-content: space-around;
                    margin-top: 20px;
                }
                .weather-item {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                }
                .weather-icon {
                    width: 50px;
                    height: 50px;
                    margin-bottom: 10px;
                }
                .weather-label {
                    font-size: 0.9rem;
                    color: #666;
                    margin-bottom: 5px;
                }
                .weather-value {
                    font-size: 1.2rem;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="weather-card">
                <div class="weather-header">
                    <div>
                        <h2>Current Weather</h2>
                        <p>{{ current_date }}</p>
                    </div>
                    <p class="temperature">{{ temperature }}°C</p>
                </div>
            
                <div class="weather-details">
                    <div class="weather-item">
                        <svg class="weather-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 3V4M12 20V21M4 12H3M21 12H20M18.364 5.636L17.657 6.343M6.343 17.657L5.636 18.364M6.343 6.343L5.636 5.636M18.364 18.364L17.657 17.657" stroke="#ff9500" stroke-width="2" stroke-linecap="round"/>
                            <circle cx="12" cy="12" r="4" fill="#ff9500"/>
                        </svg>
                        <span class="weather-label">Temperature</span>
                        <span class="weather-value">{{ temperature }}°C</span>
                    </div>
                
                    <div class="weather-item">
                        <svg class="weather-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M7 16.8C7 18.5673 8.43269 20 10.2 20C11.9673 20 13.4 18.5673 13.4 16.8C13.4 15.0327 10.2 12 10.2 12C10.2 12 7 15.0327 7 16.8Z" fill="#4299e1"/>
                            <path d="M15.4 13.2C15.4 14.1941 16.2059 15 17.2 15C18.1941 15 19 14.1941 19 13.2C19 12.2059 17.2 10 17.2 10C17.2 10 15.4 12.2059 15.4 13.2Z" fill="#4299e1"/>
                            <path d="M13.5 7.2C13.5 8.19411 14.3059 9 15.3 9C16.2941 9 17.1 8.19411 17.1 7.2C17.1 6.20589 15.3 4 15.3 4C15.3 4 13.5 6.20589 13.5 7.2Z" fill="#4299e1"/>
                            <path d="M7 16.8C7 18.5673 8.43269 20 10.2 20C11.9673 20 13.4 18.5673 13.4 16.8C13.4 15.0327 10.2 12 10.2 12C10.2 12 7 15.0327 7 16.8Z" stroke="#4299e1" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M15.4 13.2C15.4 14.1941 16.2059 15 17.2 15C18.1941 15 19 14.1941 19 13.2C19 12.2059 17.2 10 17.2 10C17.2 10 15.4 12.2059 15.4 13.2Z" stroke="#4299e1" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M13.5 7.2C13.5 8.19411 14.3059 9 15.3 9C16.2941 9 17.1 8.19411 17.1 7.2C17.1 6.20589 15.3 4 15.3 4C15.3 4 13.5 6.20589 13.5 7.2Z" stroke="#4299e1" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        <span class="weather-label">Humidity</span>
                        <span class="weather-value">{{ humidity }}%</span>
                    </div>
                
                    <div class="weather-item">
                        <svg class="weather-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M7 13C7 13 8.5 9 12 9C15.5 9 17 13 17 13" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M5 18L5.1 18M12 18L12.1 18M19 18L19.1 18M7 21L7.1 21M12 21L12.1 21M17 21L17.1 21" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path fill-rule="evenodd" clip-rule="evenodd" d="M12 9C8.5 9 7 13 7 13H17C17 13 15.5 9 12 9Z" fill="#3b82f6" fill-opacity="0.3"/>
                        </svg>
                        <span class="weather-label">Precipitation</span>
                        <span class="weather-value">{{ precipitation }} mm</span>
                    </div>
                </div>
            </div>
        """
    
        # Create a template object
        template = jinja2.Template(template_str)
    
        # Get current date
        from datetime import datetime
        current_date = datetime.now().strftime("%A, %B %d, %Y")
    
        # Render the template with the provided data
        rendered_html = template.render(
            temperature=round(temperature, 1),
            humidity=round(humidity, 1),
            precipitation=round(precipitation, 1),
            current_date=current_date
        )
    
        return rendered_html
    return (render_weather_template,)


if __name__ == "__main__":
    app.run()
