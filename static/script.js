const inputField = document.getElementById('location');
const suggestionsList = document.getElementById('suggestions');
const locationForm = document.getElementById('locationForm');

const WEATHER_API_KEY = '8017e762ac1349198ae120100242612';
const WEATHER_API_URL = 'https://api.weatherapi.com/v1/current.json';

// Handle location input and suggestions
inputField.addEventListener('input', async () => {
    const query = inputField.value.trim();

    if (query.length > 2) {
        const response = await fetch(`https://api.opencagedata.com/geocode/v1/json?q=${query}&key=6cb8fd2d1e394d2aa8ca1fb5cb122d22`);
        const data = await response.json();

        console.log("Location Suggestions Data:", data); // Log fetched data for suggestions

        suggestionsList.innerHTML = '';
        if (data.results && data.results.length > 0) {
            data.results.forEach(location => {
                const listItem = document.createElement('li');
                listItem.textContent = location.formatted;

                listItem.addEventListener('click', () => {
                    inputField.value = location.formatted;
                    suggestionsList.classList.add('d-none');
                });

                suggestionsList.appendChild(listItem);
            });
            suggestionsList.classList.remove('d-none');
        } else {
            suggestionsList.classList.add('d-none');
        }
    } else {
        suggestionsList.classList.add('d-none');
    }
});

// Hide suggestions list when clicking outside
document.addEventListener('click', (event) => {
    if (!inputField.contains(event.target) && !suggestionsList.contains(event.target)) {
        suggestionsList.classList.add('d-none');
    }
});

// Fetch weather data and send it to Flask
locationForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const location = inputField.value;

    try {
        const response = await fetch(`${WEATHER_API_URL}?key=${WEATHER_API_KEY}&q=${encodeURIComponent(location)}&aqi=yes`);
        const weatherData = await response.json();

        console.log("Weather Data:", weatherData); // Log weather data received from API

        if (weatherData.error) {
            throw new Error(weatherData.error.message);
        }

        const airQualityData = {
            temperature: weatherData.current.temp_c,
            humidity: weatherData.current.humidity,
            pm25: weatherData.current.air_quality.pm2_5,
            pm10: weatherData.current.air_quality.pm10,
            no2: weatherData.current.air_quality.no2,
            so2: weatherData.current.air_quality.so2,
            co: weatherData.current.air_quality.co,
        };

        console.log("Air Quality Data to be sent to Flask:", airQualityData); // Log data to be sent to Flask

        // Send data to Flask
        const flaskResponse = await fetch('/predictdata', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(airQualityData),
        });

        const prediction = await flaskResponse.json();

        console.log("Prediction Response from Flask:", prediction); // Log prediction response from Flask

        // Display prediction result in the UI
        const resultDiv = document.getElementById('responseDiv');
        resultDiv.innerHTML = `<h4>Prediction: ${prediction.result}</h4>`;
    } catch (error) {
        console.error('Error fetching or sending data:', error);
        const resultDiv = document.getElementById('responseDiv');
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
});
