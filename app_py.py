{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/abrahamtobi96/maintenance-app/blob/main/app_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bk5usA-mpbqF",
        "outputId": "b0671051-0e0e-4719-b2a2-dcbd7de8dec2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m44.3/44.3 kB\u001b[0m \u001b[31m713.9 kB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m9.8/9.8 MB\u001b[0m \u001b[31m30.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m22.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m79.1/79.1 kB\u001b[0m \u001b[31m3.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ],
      "source": [
        "!pip install streamlit --quiet"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install joblib==1.3.2"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nc0pZN4lhWtZ",
        "outputId": "79260315-2965-4856-a357-7e7bd63890e2"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting joblib==1.3.2\n",
            "  Downloading joblib-1.3.2-py3-none-any.whl.metadata (5.4 kB)\n",
            "Downloading joblib-1.3.2-py3-none-any.whl (302 kB)\n",
            "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/302.2 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[91m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[90m╺\u001b[0m\u001b[90m━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m153.6/302.2 kB\u001b[0m \u001b[31m4.5 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m302.2/302.2 kB\u001b[0m \u001b[31m5.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: joblib\n",
            "  Attempting uninstall: joblib\n",
            "    Found existing installation: joblib 1.4.2\n",
            "    Uninstalling joblib-1.4.2:\n",
            "      Successfully uninstalled joblib-1.4.2\n",
            "Successfully installed joblib-1.3.2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kw5C_1Js-Vn-",
        "outputId": "494f4042-2a93-437b-dab6-219a0ec75df1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import joblib\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "# ---- Set page configuration (must be first Streamlit command) ----\n",
        "st.set_page_config(page_title='Maintenance Cost Prediction', page_icon=':bar_chart:', layout='wide')\n",
        "\n",
        "# ---- Cache the model loading to avoid reloading every run ----\n",
        "@st.cache_resource\n",
        "def load_models():\n",
        "    mc_model_data = joblib.load('/content/mc_model.pkl')\n",
        "    dtf_model_data = joblib.load('/content/dtf_model.pkl')\n",
        "\n",
        "    mc_model = mc_model_data['mc_model']           # Extract from dict\n",
        "    dtf_model = dtf_model_data['dtf_model_dt']     # Extract from dict\n",
        "\n",
        "    return mc_model, dtf_model\n",
        "\n",
        "# ---- Load Models ----\n",
        "mc_model, dtf_model = load_models()\n",
        "\n",
        "# ---- Page Title ----\n",
        "st.markdown(\"<h1 style='text-align: center; color: red;'>Maintenance Cost Prediction</h1>\", unsafe_allow_html=True)\n",
        "st.subheader('Maintenance Cost Prediction')\n",
        "\n",
        "# ---- Sidebar Input ----\n",
        "st.sidebar.header('Equipment Parameters')\n",
        "input_data = {\n",
        "    'Temperature': st.sidebar.number_input(\"Temperature (°C)\", min_value=0.00, max_value=100.00, value=50.00, help=\"Enter the equipment's operating temperature\"),\n",
        "    'Pressure': st.sidebar.number_input('Pressure (%)', min_value=0.00, max_value=100.00, value=50.00, help=\"Enter the equipment's operating pressure\"),\n",
        "    'Vibration': st.sidebar.number_input('Vibration', min_value=0.00, max_value=5.00, value=2.00, help=\"Enter the equipment's vibration\"),\n",
        "    'Humidity': st.sidebar.number_input('Humidity', min_value=0.00, max_value=2.00, value=1.00, help=\"Enter the equipment's humidity\"),\n",
        "    'Flow_Rate': st.sidebar.number_input('Flow Rate', min_value=0.00, max_value=15.00, value=10.00, help=\"Enter the equipment's flow rate\"),\n",
        "    'Power_Consumption': st.sidebar.number_input('Power Consumption', min_value=0.00, max_value=500.00, value=200.00, help=\"Enter the equipment's power consumption\"),\n",
        "    'Oil_Level': st.sidebar.number_input('Oil Level', min_value=0.00, max_value=1.00, value=0.50, help=\"Enter the equipment's oil level\"),\n",
        "    'Voltage': st.sidebar.number_input('Voltage', min_value=100.00, max_value=300.00, value=200.00, help=\"Enter the equipment's voltage\"),\n",
        "    'Production_Volume': st.sidebar.number_input('Production Volume', min_value=0, max_value=500, value=200, help=\"Enter the equipment's production volume\"),\n",
        "    'Planned_Downtime_Hours': st.sidebar.number_input('Planned Downtime Hours', min_value=0, max_value=24, value=8, help=\"Enter the equipment's planned downtime hours\"),\n",
        "    'Shifts_Per_Day': st.sidebar.number_input('Shifts Per Day', min_value=1, max_value=3, value=2, help=\"Enter the number of shifts per day\"),\n",
        "    'Production_Days_Per_Week': st.sidebar.number_input('Production Days Per Week', min_value=1, max_value=7, value=3, help=\"Enter the number of production days per week\"),\n",
        "    'Maintenance_Type_Corrective': st.sidebar.checkbox('Maintenance Type Corrective', help=\"Select if the maintenance type is corrective\"),\n",
        "    'Maintenance_Type_Preventive': st.sidebar.checkbox('Maintenance Type Preventive', help=\"Select if the maintenance type is preventive\"),\n",
        "    'Failure_Cause_Electrical_Failure': st.sidebar.checkbox('Failure Cause Electrical Failure', help=\"Select if the failure cause is electrical failure\"),\n",
        "    'Failure_Cause_Mechanical_Failure': st.sidebar.checkbox('Failure Cause Mechanical Failure', help=\"Select if the failure cause is mechanical failure\"),\n",
        "    'Failure_Cause_Sensor_Malfunction': st.sidebar.checkbox('Failure Cause Sensor Malfunction', help=\"Select if the failure cause is sensor malfunction\"),\n",
        "\n",
        "}\n",
        "\n",
        "\n",
        "# ---- Preprocess Input ----\n",
        "input_df = pd.DataFrame(input_data, index=[0])\n",
        "\n",
        "# ---- Make Predictions ----\n",
        "if st.button('Predict'):\n",
        "    # Code to execute when the button is clicked\n",
        "    # This is where you will call your prediction functions\n",
        "    try:\n",
        "        maintenance_cost = mc_model.predict(input_df)[0]\n",
        "        days_till_failure = dtf_model.predict(input_df)[0]\n",
        "\n",
        "        # Display Results\n",
        "        st.success(f\"Predicted Maintenance Cost: ${maintenance_cost:.2f}\")\n",
        "        st.warning(f\"Predicted Days Till Failure: {days_till_failure:.0f} days\")\n",
        "\n",
        "    except Exception as e:\n",
        "        st.error(f\"Prediction Error: {e}\")\n",
        "        st.warning(\"Please check your input values and try again.\")\n",
        "\n",
        "# ---- Custom Footer ----\n",
        "st.markdown(\"\"\"\n",
        "    <style>\n",
        "    .footer {\n",
        "        position: fixed;\n",
        "        left: 0;\n",
        "        bottom: 0;\n",
        "        width: 100%;\n",
        "        background-color: #4CAF50;\n",
        "        color: white;\n",
        "        text-align: center;\n",
        "        padding: 10px;\n",
        "    }\n",
        "    .footer a {\n",
        "        color: white;\n",
        "        text-decoration: none;\n",
        "        margin: 0 10px;\n",
        "    }\n",
        "    </style>\n",
        "    <div class=\"footer\">\n",
        "        Developed by Tobi  - ©2025\n",
        "        <br>\n",
        "        <a href=\"https://www.linkedin.com/in/Tobi_Oluwasola/\" target=\"_blank\">LinkedIn</a>\n",
        "        <a href=\"https://github.com/abrahamtobi96\" target=\"_blank\">GitHub</a>\n",
        "    </div>\n",
        "\"\"\", unsafe_allow_html=True)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tUso4GNztsLt",
        "outputId": "40d0f88f-3ffb-4372-9fe8-aca175f262d0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "34.56.90.159\n"
          ]
        }
      ],
      "source": [
        "!wget -q -O - ipv4.icanhazip.com"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PNK6kIbBdIxL",
        "outputId": "3f629036-fd78-46c0-f2ac-04a3aa081c71"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[1G\u001b[0K⠙\u001b[1G\u001b[0K⠹\u001b[1G\u001b[0K⠸\u001b[1G\u001b[0K⠼\u001b[1G\u001b[0K⠴\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.56.90.159:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[1G\u001b[0K⠦\u001b[1G\u001b[0K\u001b[1G\u001b[0JNeed to install the following packages:\n",
            "localtunnel@2.0.2\n",
            "Ok to proceed? (y) \u001b[20Gy\n",
            "\n",
            "\u001b[1G\u001b[0K⠙\u001b[1G\u001b[0K⠹\u001b[1G\u001b[0K⠸\u001b[1G\u001b[0K⠼\u001b[1G\u001b[0K⠴\u001b[1G\u001b[0K⠦\u001b[1G\u001b[0K⠧\u001b[1G\u001b[0K⠇\u001b[1G\u001b[0K⠏\u001b[1G\u001b[0K⠋\u001b[1G\u001b[0K⠙\u001b[1G\u001b[0K⠹\u001b[1G\u001b[0K⠸\u001b[1G\u001b[0K⠼\u001b[1G\u001b[0K⠴\u001b[1G\u001b[0K⠦\u001b[1G\u001b[0K⠧\u001b[1G\u001b[0K⠇\u001b[1G\u001b[0K⠏\u001b[1G\u001b[0K⠋\u001b[1G\u001b[0Kyour url is: https://silent-crabs-film.loca.lt\n"
          ]
        }
      ],
      "source": [
        "!streamlit run app.py & npx localtunnel --port 8501"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "12iru9WL9QOJHGSlUsrzjgaJS71WfyoSg",
      "authorship_tag": "ABX9TyP2tDaGpe9a/TL3W6ZbG3YN",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}