from http.client import HTTPException
from fastapi import FastAPI, Request, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import nbformat
from fastapi import FastAPI, Request, Query
from fastapi import Query, Path



templates = Jinja2Templates(directory="templates")


# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load historical data for comparisons
historical_data = pd.read_excel("Internships.xlsx", engine="openpyxl")

# Drop unnecessary columns
columns_to_drop = ['Last modified time', 'Email', 'Name', 'Completion time', 'Start time']
historical_data = historical_data.drop(columns=columns_to_drop, errors='ignore')


# Initialize FastAPI
app = FastAPI()

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Endpoint to display the HTML form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/predict/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/statistics/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("statistics.html", {"request": request})
@app.get("/insights/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("insights.html", {"request": request})






@app.get("/insights/upload/", response_class=HTMLResponse)
async def render_upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# Encoding mappings
gender_mapping = {"Male": 0, "Female": 1}
year_of_studies_mapping = {"2nd year": 0, "3rd year": 1, "4th year": 2, "5th year or more": 3}
field_of_study_mapping = {
    "Finance & Business": 0,
    "Education & Teaching": 1,
    "Engineering": 2,
    "Architecture": 3,
    "Health field": 4,
    "Creative Arts": 5,
    "Other": 6
}
career_support_mapping = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3, "Very good": 4}



# Ensure encoding for categorical fields in historical_data
categorical_columns = {
    "What is your gender?\n": {"Male": 0, "Female": 1},
    "What is your year of studies?\n": {"2nd year": 0, "3rd year": 1, "4th year": 2, "5th year or more": 3},
    "What is your field of study?": {
        "Finance & Business": 0,
        "Education & Teaching": 1,
        "Engineering": 2,
        "Architecture": 3,
        "Health field": 4,
        "Creative Arts": 5,
        "Other": 6
    },
    "How would you rate the adequacy of your university's career support?\n": {
        "Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3, "Very good": 4
    }
}
for col, mapping in categorical_columns.items():
    if col in historical_data:
        historical_data[col] = historical_data[col].map(mapping)



# Feature columns
feature_columns = [
    "What is your gender?\n",
    "What is your year of studies?\n",
    "What is your field of study?",
    "How many internships have you completed?\n",
    "On a scale of 1 to 10, how important do you believe internships are in building your professional network? ",
    "How would you rate the adequacy of your university's career support?\n",
    "How many professional development activities have you attended in the past year (including workshops, seminars, and networking events)?"
]

# Endpoint to handle form submission and make a prediction

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: str = Form(...),
    year_of_studies: str = Form(...),
    field_of_study: str = Form(...),
    internships_completed: int = Form(...),
    network_importance: int = Form(...),
    career_support: str = Form(...),
    professional_activities: int = Form(...)
):
    try:
        # Encode user inputs
        gender_encoded = gender_mapping.get(gender, -1)
        year_of_studies_encoded = year_of_studies_mapping.get(year_of_studies, -1)
        field_of_study_encoded = field_of_study_mapping.get(field_of_study, -1)
        career_support_encoded = career_support_mapping.get(career_support, -1)

        # Prepare user data
        user_data = pd.DataFrame([{
            "What is your gender?\n": gender_encoded,
            "What is your year of studies?\n": year_of_studies_encoded,
            "What is your field of study?": field_of_study_encoded,
            "How many internships have you completed?\n": internships_completed,
            "On a scale of 1 to 10, how important do you believe internships are in building your professional network? ": network_importance,
            "How would you rate the adequacy of your university's career support?\n": career_support_encoded,
            "How many professional development activities have you attended in the past year (including workshops, seminars, and networking events)?": professional_activities
        }])

        # Predict probability for the user
        user_probability = model.predict_proba(user_data[feature_columns])[0][1]

        # Field distribution and probabilities
        field_counts = historical_data["What is your field of study?"].value_counts()
        field_probs = []

        for field, field_code in field_of_study_mapping.items():
            if field_code in field_counts.index:
                count = field_counts[field_code]
                field_data = historical_data[historical_data["What is your field of study?"] == field_code]
                avg_probability = model.predict_proba(field_data[feature_columns])[:, 1].mean() * 100
            else:
                count = 0
                avg_probability = 0
            field_probs.append((field, count, avg_probability))

        # Sort fields for better visualization
        field_probs.sort(key=lambda x: x[0])  # Sort alphabetically by field name
        fields = [fp[0] for fp in field_probs]
        counts = [fp[1] for fp in field_probs]
        probabilities = [fp[2] for fp in field_probs]

        # Create the graph
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart for counts
        bar_width = 0.4
        bar_positions = range(len(fields))
        bars = ax1.bar(bar_positions, counts, color="skyblue", alpha=0.7, label="Field Count", width=bar_width)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_xlabel("Fields of Study", fontsize=12)
        ax1.set_xticks(bar_positions)
        ax1.set_xticklabels(fields, rotation=45)

        # Line chart for probabilities
        ax2 = ax1.twinx()
        #ax2.plot(bar_positions, probabilities, color="green", marker="o", label="Field Probabilities")

        # Highlight user probability
        if field_of_study in fields:
            user_field_index = fields.index(field_of_study)
            ax2.scatter(user_field_index, user_probability * 100, color="orange", label="User Probability", zorder=5)
            ax2.axhline(user_probability * 100, color="orange", linestyle="dashed", linewidth=1)

        ax2.set_ylabel("Probability (%)", fontsize=12)

        # Add counts above bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{count}", ha="center", va="bottom")

        # Add percentages above probabilities
      #  for i, prob in enumerate(probabilities):
       #     ax2.text(bar_positions[i], prob + 1, f"{prob:.1f}%", ha="center", va="bottom", color="green")

        # Set title and legend
        ax1.set_title("Field Distribution with Internship Probabilities", fontsize=14)
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

        # Convert graph to Base64 for HTML rendering
        graph_base64 = _convert_plot_to_base64(fig)


        # Predict probability for the user
        user_probability = model.predict_proba(user_data[feature_columns])[0][1]

        # Graph 1 (Already Implemented Logic for Field Distribution)
        # (Existing code for Graph 1 remains here.)








        # Graph 2: Internships Completed by Year
        avg_internships_by_year = (
            historical_data.groupby("What is your year of studies?\n")["How many internships have you completed?\n"]
            .mean()
            .sort_index()
        )

        # Map the years to their corresponding labels
        year_labels = {
            0: "2nd year",
            1: "3rd year",
            2: "4th year",
            3: "5th year or more",
        }

        # Apply mapping to display meaningful labels on the X-axis
        mapped_years = [year_labels.get(year, "Unknown") for year in avg_internships_by_year.index.tolist()]
        avg_internships = avg_internships_by_year.values.tolist()

        # Check the user year and generate the appropriate message
        user_year_index = year_of_studies_mapping.get(year_of_studies, -1)

        user_message = ""
        if user_year_index in avg_internships_by_year.index:
            avg_for_user_year = avg_internships_by_year[user_year_index]
            if internships_completed < avg_for_user_year:
                user_message = "You should do more internships because you are below the average for your year."
            else:
                user_message = "Great! You are above or at the average for internships in your year."
        else:
            user_message = "No average data available for your year."

        # Create the graph for Graph 2
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Bar chart for average internships by year
        bars2 = ax2.bar(
            mapped_years,
            avg_internships,
            color="skyblue",
            alpha=0.7,
            label="Average Internships by Year",
        )
        ax2.set_xlabel("Year of Study", fontsize=12)
        ax2.set_ylabel("Average Internships Completed", fontsize=12)
        ax2.set_title("Graph 2: Internships Completed by Year", fontsize=14)

        # Highlight user's internship count
        if user_year_index in avg_internships_by_year.index:
            user_year_position = list(avg_internships_by_year.index).index(user_year_index)
            ax2.scatter(
                user_year_position,
                internships_completed,
                color="orange",
                label="Your Internships",
                zorder=5,
            )
            ax2.axhline(
                y=internships_completed,
                color="orange",
                linestyle="dashed",
                linewidth=1,
                label="User's Internship Count",
            )

            # Annotate the user's point
            ax2.text(
                user_year_position,
                internships_completed + 0.1,
                f"Your Count: {internships_completed}",
                color="orange",
                ha="center",
                va="bottom",
            )

        # Add labels above bars
        for bar, avg in zip(bars2, avg_internships):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{avg:.1f}",
                ha="center",
                va="bottom",
            )

        # Convert graph to Base64 for HTML rendering
        graph2_base64 = _convert_plot_to_base64(fig2)

        

        # Convert graph to Base64 for HTML rendering
        graph2_base64 = _convert_plot_to_base64(fig2)
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "probability": f"{round(user_probability * 100, 2)}%",
                "graph1": f"data:image/png;base64,{graph_base64}",
                "graph2": f"data:image/png;base64,{graph2_base64}"   # Include Graph 2
            } 

        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details:\n{error_details}")
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})


# Helper function for converting plot to Base64 remains unchanged.


def _convert_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return plot_base64


@app.get("/statistics/", response_class=HTMLResponse)
async def show_statistics(request: Request):
    try:
        # Load user data
        file_path = "user_data.xlsx"
        if not os.path.exists(file_path):
            return {"error": "No data available for statistics."}

        user_data = pd.read_excel(file_path, engine="openpyxl")

        # Generate a histogram for probabilities
        if "Probability" not in user_data.columns:
            user_data["Probability"] = model.predict_proba(user_data.drop(columns=["Probability"]))[:, 1]

        plt.figure(figsize=(8, 6))
        plt.hist(user_data["Probability"] * 100, bins=10, color="skyblue", edgecolor="black")
        plt.title("Distribution of Probabilities")
        plt.xlabel("Probability (%)")
        plt.ylabel("Frequency")
        plt.savefig("static/probability_distribution.png")  # Save the plot
        plt.close()

        # Render the statistics page
        return templates.TemplateResponse("statistics.html", {"request": request})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details:\n{error_details}")
        return {"error": f"An internal error occurred: {str(e)}"}
#
#
#
#
#the merge beggin
#
# 
#  Directories for uploads and static files
UPLOAD_FOLDER = "./uploads"
IMAGE_FOLDER = "./static/imagesl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# Route to upload a notebook
@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def upload_notebook(request: Request, notebook: UploadFile):
    if not notebook.filename.endswith(".ipynb"):
        raise HTTPException(status_code=400, detail="Please upload a valid .ipynb file.")
    
    notebook_path = os.path.join(UPLOAD_FOLDER, notebook.filename)
    with open(notebook_path, "wb") as f:
        f.write(await notebook.read())

    try:
        process_notebook(notebook_path)
        return RedirectResponse(url="/visualizations", status_code=302)
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"Error processing the notebook: {str(e)}"})
# 
# 
# Helper function to convert plots to Base64
def _convert_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return plot_base64

# Handle notebook upload and process insights
@app.post("/insights/upload/", response_class=HTMLResponse)
async def upload_notebook(request: Request, notebook: UploadFile):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, notebook.filename)
        with open(file_path, "wb") as f:
            f.write(await notebook.read())

        # Process the notebook and generate visualizations
        process_notebook(file_path)

        # Redirect to visualizations page
        return RedirectResponse(url="/visualizations/", status_code=302)
    except Exception as e:
        return templates.TemplateResponse(
            "error.html", {"request": request, "message": f"An error occurred: {str(e)}"}
        )
    
def process_uploaded_file(file_path):
    # Example processing logic
    df = pd.read_excel(file_path, engine="openpyxl")

    # Extract insights
    field_insight = f"The most common field is {df['Field'].mode()[0]}."
    engagement_insight = f"The average professional activities per user are {df['Professional Activities'].mean():.2f}."
    internship_insight = f"The average internships completed are {df['Internships'].mean():.2f}."

    return {
        "field": field_insight,
        "engagement": engagement_insight,
        "internship": internship_insight
    }


# Endpoint to display generated visualizations
@app.get("/visualizations/", response_class=HTMLResponse)
async def visualizations(request: Request):
    histograms = []
    for image_file in os.listdir(IMAGE_FOLDER):
        if image_file.endswith(".png"):
            histograms.append({
                "title": image_file.replace("_", " ").replace(".png", "").capitalize(),
                "image": f"imagesl/{image_file}",
                "description": f"This visualization shows {image_file.replace('_', ' ').replace('.png', '').lower()}."
            })
    return templates.TemplateResponse("visualizations.html", {"request": request, "histograms": histograms})

# 
# 
# Function to process the uploaded notebook and generate visualizations
def process_notebook(notebook_path: str):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Simulate data processing and create visualizations
    df = pd.DataFrame({
        "Category": ["A", "B", "C", "D"],
        "Values": [10, 20, 15, 25]
    })

    # Generate a sample histogram
    plt.bar(df["Category"], df["Values"])
    plt.title("Sample Histogram")
    plt.savefig(os.path.join(IMAGE_FOLDER, "sample_histogram.png"))
    plt.close()
# Zoom into a specific visualization

@app.get("/zoom/{image:path}", response_class=HTMLResponse)
async def zoom(
    request: Request,
    image: str = Path(..., description="Path to the image"),
    title: str = Query("Zoomed Image", description="Title for the visualization"),
    description: str = Query("Detailed view of the selected visualization.", description="Description of the visualization")
):
    return templates.TemplateResponse(
        "zoom.html",
        {
            "request": request,
            "image": f"/static/{image}",  # Correctly point to the static folder
            "title": title,
            "description": description,
        }
    )

