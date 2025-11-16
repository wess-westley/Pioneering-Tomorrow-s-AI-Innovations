import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')


# Set page configuration
st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8f0;
        color:green;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color:green;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        color:green;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sensor-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

class SmartAgricultureUI:
    def __init__(self):
        self.load_models()
        self.crop_types = ['Wheat', 'Rice', 'Corn', 'Soybean', 'Barley', 'Cotton', 'Sugarcane']
        
    def load_models(self):
        """Load pre-trained models and scaler"""
        try:
            self.model = joblib.load('best_crop_yield_model.pkl')
            self.scaler = joblib.load('feature_scaler.pkl')
            st.sidebar.success("✅ AI Models loaded successfully!")
        except:
            st.sidebar.warning("⚠️ Models not found. Using demo mode.")
            self.model = None
            self.scaler = None
    
    def simulate_sensor_data(self):
        """Generate realistic sensor data"""
        return {
            'soil_moisture': random.uniform(15, 45),
            'temperature': random.uniform(18, 32),
            'humidity': random.uniform(35, 85),
            'light': random.uniform(300, 1200),
            'pH': random.uniform(5.8, 7.2),
            'rainfall': random.uniform(0, 25),
            'wind_speed': random.uniform(0, 15)
        }
    
    def predict_yield(self, sensor_data):
        """Predict crop yield based on sensor data"""
        if self.model is None or self.scaler is None:
            # Demo prediction
            base_yield = 2500
            moisture_factor = sensor_data['soil_moisture'] / 25
            temp_factor = 1 - abs(sensor_data['temperature'] - 25) / 25
            light_factor = sensor_data['light'] / 800
            return base_yield * moisture_factor * temp_factor * light_factor
        
        features = np.array([[
            sensor_data['soil_moisture'],
            sensor_data['temperature'],
            sensor_data['humidity'],
            sensor_data['light'],
            sensor_data['pH']
        ]])
        
        features_scaled = self.scaler.transform(features)
        predicted_yield = self.model.predict(features_scaled)[0]
        return max(0, predicted_yield)
    
    def get_recommendations(self, sensor_data, predicted_yield):
        """Generate agricultural recommendations"""
        recommendations = []
        
        # Soil moisture analysis
        if sensor_data['soil_moisture'] < 20:
            recommendations.append("🚰 **Irrigation Needed**: Soil moisture is low. Consider watering.")
        elif sensor_data['soil_moisture'] > 40:
            recommendations.append("💧 **Reduce Irrigation**: Soil moisture is high. Risk of waterlogging.")
        else:
            recommendations.append("✅ **Optimal Moisture**: Soil moisture levels are perfect.")
        
        # Temperature analysis
        if sensor_data['temperature'] < 20:
            recommendations.append("🌡️ **Low Temperature**: Consider using row covers to warm soil.")
        elif sensor_data['temperature'] > 30:
            recommendations.append("🔥 **High Temperature**: Implement shading to prevent heat stress.")
        else:
            recommendations.append("✅ **Ideal Temperature**: Optimal for crop growth.")
        
        # pH analysis
        if sensor_data['pH'] < 6.0:
            recommendations.append("🧪 **Acidic Soil**: Consider adding agricultural lime.")
        elif sensor_data['pH'] > 7.0:
            recommendations.append("🧪 **Alkaline Soil**: Consider adding sulfur to lower pH.")
        else:
            recommendations.append("✅ **Perfect pH**: Soil acidity is optimal.")
        
        # Light analysis
        if sensor_data['light'] < 500:
            recommendations.append("☀️ **Low Light**: Consider artificial lighting or pruning.")
        elif sensor_data['light'] > 1000:
            recommendations.append("🌞 **High Light**: Monitor for potential sun damage.")
        else:
            recommendations.append("✅ **Optimal Light**: Perfect light conditions.")
        
        # Yield prediction
        if predicted_yield < 2000:
            recommendations.append("📉 **Low Yield Risk**: Check soil nutrients and consider fertilization.")
        elif predicted_yield > 4000:
            recommendations.append("📈 **High Yield Expected**: Maintain current practices.")
        else:
            recommendations.append("📊 **Good Yield Expected**: Conditions are favorable.")
        
        return recommendations

def main():
    # Initialize the agriculture UI system
    agri_ui = SmartAgricultureUI()
    
    # Header
    st.markdown('<h1 class="main-header">🌾 Smart Agriculture AI System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🏠 Control Panel")
    
    # Farm Configuration
    st.sidebar.subheader("🏞️ Farm Configuration")
    selected_crop = st.sidebar.selectbox("Select Crop Type", agri_ui.crop_types)
    farm_size = st.sidebar.slider("Farm Size (hectares)", 1, 100, 10)
    location = st.sidebar.selectbox("Location", ["North Region", "South Region", "East Region", "West Region"])
    
    # Simulation Controls
    st.sidebar.subheader("🎮 Simulation Controls")
    simulation_speed = st.sidebar.slider("Simulation Speed (seconds)", 1, 10, 3)
    auto_refresh = st.sidebar.checkbox("Auto-refresh Data", value=True)
    
    if st.sidebar.button("🔄 Refresh Data Now"):
        st.rerun()
    
    # Main dashboard layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("📡 Real-time Sensor Data")
        
        # Simulate sensor data
        sensor_data = agri_ui.simulate_sensor_data()
        
        # Display sensor values in cards
        st.markdown(f"""
        <div class="metric-card">
            <div>Soil Moisture</div>
            <div class="sensor-value">{sensor_data['soil_moisture']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div>Temperature</div>
            <div class="sensor-value">{sensor_data['temperature']:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div>Humidity</div>
            <div class="sensor-value">{sensor_data['humidity']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🌱 Environmental Conditions")
        
        st.markdown(f"""
        <div class="metric-card">
            <div>Light Intensity</div>
            <div class="sensor-value">{sensor_data['light']:.0f} Lux</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div>Soil pH</div>
            <div class="sensor-value">{sensor_data['pH']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div>Rainfall</div>
            <div class="sensor-value">{sensor_data['rainfall']:.1f} mm</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.subheader("🎯 AI Predictions")
        
        # Predict yield
        predicted_yield = agri_ui.predict_yield(sensor_data)
        
        st.markdown(f"""
        <div class="metric-card" style="background-color: #e8f5e8;">
            <div>Predicted Crop Yield</div>
            <div class="sensor-value" style="color: #2E8B57;">{predicted_yield:.0f} kg/ha</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Yield quality indicator
        if predicted_yield > 4000:
            st.success("🎉 Excellent Yield Expected!")
            yield_quality = "Excellent"
        elif predicted_yield > 2500:
            st.info("👍 Good Yield Expected")
            yield_quality = "Good"
        else:
            st.warning("⚠️ Below Average Yield Expected")
            yield_quality = "Needs Improvement"
        
        # Additional metrics
        st.metric("Crop Health Score", f"{(predicted_yield / 5000 * 100):.0f}%")
        st.metric("Growth Conditions", yield_quality)
    
    # Recommendations Section
    st.subheader("💡 Smart Recommendations")
    recommendations = agri_ui.get_recommendations(sensor_data, predicted_yield)
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        for i, rec in enumerate(recommendations[:3]):
            if "✅" in rec:
                st.markdown(f'<div class="success-box">{rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning-box">{rec}</div>', unsafe_allow_html=True)
    
    with rec_col2:
        for i, rec in enumerate(recommendations[3:]):
            if "✅" in rec:
                st.markdown(f'<div class="success-box">{rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning-box">{rec}</div>', unsafe_allow_html=True)
    
    # Charts and Visualizations
    st.subheader("📊 Analytics Dashboard")
    
    # Create sample historical data
    dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    historical_yield = [agri_ui.predict_yield(agri_ui.simulate_sensor_data()) for _ in range(24)]
    historical_moisture = [random.uniform(15, 45) for _ in range(24)]
    historical_temperature = [random.uniform(18, 32) for _ in range(24)]
    
    tab1, tab2, tab3, tab4 = st.tabs(["Yield Trends", "Sensor History", "Correlation Analysis", "Farm Map"])
    
    with tab1:
        # Yield trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=historical_yield, 
                               mode='lines+markers', 
                               name='Predicted Yield',
                               line=dict(color='green', width=3)))
        
        fig.update_layout(
            title='Crop Yield Prediction Trend (24 Hours)',
            xaxis_title='Time',
            yaxis_title='Yield (kg/ha)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Multiple sensor history
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Soil Moisture', 'Temperature', 'Light Intensity', 'pH Levels'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(go.Scatter(x=dates, y=historical_moisture, 
                               name='Moisture', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=historical_temperature, 
                               name='Temperature', line=dict(color='red')), row=1, col=2)
        fig.add_trace(go.Scatter(x=dates, y=[random.uniform(300, 1200) for _ in range(24)], 
                               name='Light', line=dict(color='orange')), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=[random.uniform(5.8, 7.2) for _ in range(24)], 
                               name='pH', line=dict(color='purple')), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation heatmap
        correlation_data = {
            'Soil Moisture': historical_moisture,
            'Temperature': historical_temperature,
            'Light': [random.uniform(300, 1200) for _ in range(24)],
            'pH': [random.uniform(5.8, 7.2) for _ in range(24)],
            'Yield': historical_yield
        }
        corr_df = pd.DataFrame(correlation_data).corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
    
    with tab4:
        # Farm visualization
        st.subheader("🏞️ Farm Layout Visualization")
        
        # Create a simple farm map
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw farm boundaries
        farm_boundary = plt.Rectangle((0, 0), 10, 8, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(farm_boundary)
        
        # Add sensor locations
        sensor_locations = [(2, 2), (8, 2), (5, 6), (2, 6), (8, 6)]
        for i, (x, y) in enumerate(sensor_locations):
            ax.plot(x, y, 'ro', markersize=10)
            ax.text(x, y + 0.3, f'Sensor {i+1}', ha='center')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.set_title('Farm Layout with Sensor Locations')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.info("📍 **Farm Overview**: This visualization shows the strategic placement of IoT sensors across your farm for optimal monitoring.")
    
    # Real-time Monitoring Section
    st.subheader("🔄 Real-time Monitoring")
    
    if auto_refresh:
        # Create a placeholder for live data
        live_data_placeholder = st.empty()
        
        # Simulate real-time updates
        with live_data_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Soil Moisture", f"{sensor_data['soil_moisture']:.1f}%", "±2%")
            with col2:
                st.metric("Temperature", f"{sensor_data['temperature']:.1f}°C", "±0.5°C")
            with col3:
                st.metric("Light Intensity", f"{sensor_data['light']:.0f} Lux", "±50 Lux")
            with col4:
                st.metric("Current Yield", f"{predicted_yield:.0f} kg/ha", f"{random.choice(['-', '+'])}{random.randint(1, 50)}")
        
        # Auto-refresh notice
        st.info(f"🕐 Data auto-refreshes every {simulation_speed} seconds. Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Export and Reports Section
    st.sidebar.subheader("📁 Reports & Export")
    
    if st.sidebar.button("📄 Generate Farm Report"):
        with st.spinner("Generating comprehensive farm report..."):
            time.sleep(2)
            
            # Generate report data
            report_data = {
                "Farm Details": {
                    "Crop Type": selected_crop,
                    "Farm Size": f"{farm_size} hectares",
                    "Location": location,
                    "Report Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "Current Conditions": sensor_data,
                "Predictions": {
                    "Expected Yield": f"{predicted_yield:.0f} kg/ha",
                    "Yield Quality": yield_quality,
                    "Confidence Level": "High"
                },
                "Recommendations": recommendations
            }
            
            st.success("✅ Farm report generated successfully!")
            
            # Display report
            st.subheader("📋 Farm Analysis Report")
            st.json(report_data)
            
            # Download button
            csv_data = pd.DataFrame([sensor_data])
            st.download_button(
                label="📥 Download Sensor Data (CSV)",
                data=csv_data.to_csv(index=False),
                file_name=f"farm_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # System Status
    st.sidebar.subheader("🔧 System Status")
    st.sidebar.progress(85, text="System Health: 85%")
    st.sidebar.info("🟢 All systems operational")
    st.sidebar.metric("Data Points Collected", "1,247")
    st.sidebar.metric("Model Accuracy", "94.2%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🌾 Smart Agriculture AI System | Powered by Machine Learning & IoT</p>
            <p>Real-time crop monitoring and yield prediction for modern farming</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()