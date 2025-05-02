from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import requests
import json
import re  # For cleaning API response
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://clickresume.vercel.app", "http://localhost:5173"]}}, supports_credentials=True)

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# class function for ats score
class ATSMatcher:
    def __init__(self, job_data_json):
        """
        Initialize the ATSMatcher with job data from a JSON object.
        :param job_data_json: JSON object containing job data (similar to a MongoDB document).
        """
        self.job_df = pd.DataFrame(job_data_json)
        self.vectorizer = TfidfVectorizer(stop_words='english')
    # def calculate_ats_scoreMERN(self, skills, cgpa, job_role=None):
    #     """
    #     Calculates the ATS score based on skills, experience, and CGPA.
    #
    #     :param skills: List of candidate's skills.
    #     :param experience: Candidate's years of experience.
    #     :param cgpa: Candidate's CGPA.
    #     :param job_role: (Optional) Specific job role to filter jobs.
    #     :return: DataFrame with top 5 matching jobs sorted by ATS score.
    #     """
    #     if self.job_df.empty:
    #         return "No job listings available."
    #
    #     # Filter jobs by role if specified
    #     job_df = self.job_df.copy()
    #
    #     if job_df.empty:
    #         return f"No jobs found for the role '{job_role}'."
    #
    #     # Preprocess candidate skills
    #     skills_text = " ".join(skills).lower()
    #
    #     # TF-IDF Vectorization for job descriptions
    #     job_descriptions = job_df["jobDescription"].fillna("").tolist()
    #     job_descriptions.append(skills_text)  # Append candidate's skills as a pseudo-description
    #     tfidf_matrix = self.vectorizer.fit_transform(job_descriptions)
    #
    #     # Compute similarity scores
    #     similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    #
    #     # Normalize similarity scores (scale between 0-100)
    #     similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min() + 1e-5) * 100
    #
    #     # Experience and CGPA matching (Scaled for ATS Score Calculation)  # Penalizing large differences
    #     job_df["CGPA Score"] = 100 - abs(job_df["cgpa"] - cgpa) * 20
    #     job_df["CGPA Score"] = job_df["CGPA Score"].clip(0, 100)
    #
    #     # Final ATS Score Calculation (Weighted)
    #     job_df["ATS Score"] = (0.6 * similarity_scores) + (0.01 * job_df["CGPA Score"])
    #
    #     # Sort and return top 5 matches
    #     top_matches = job_df[["_id", "jobRole", "ATS Score"]].sort_values(
    #         by="ATS Score", ascending=False
    #     )
    #
    #     return top_matches

    def calculate_ats_scoreMERN(self, user_skills, user_cgpa):
        """
        Calculates ATS score using cosine similarity between user skills and job descriptions+skills.

        :param user_skills: List of skills extracted from the user's resume.
        :param user_cgpa: CGPA extracted from the user's resume.
        :return: DataFrame with top 5 job matches sorted by ATS score.
        """
        if self.job_df.empty:
            return "No job listings available."

        job_df = self.job_df.copy()

        # Prepare combined job text for each row
        def combine_job_text(row):
            desc = str(row.get("jobDescription", "")) # here jobDescription to description
            skills = row.get("requiredSkills", []) # here requiredSkills to skills
            skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills)
            return f"{desc} {skills_str}".lower()

        job_df["combinedText"] = job_df.apply(combine_job_text, axis=1)
        print(job_df)

        # Prepare user profile text (just skills, because CGPA is not used for similarity here)
        user_profile_text = ", ".join([skill.strip().lower() for skill in user_skills])

        # Vectorize using TF-IDF
        documents = job_df["combinedText"].tolist() + [user_profile_text]
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(documents)

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        print(similarity_scores)

        # Normalize similarity scores between 0 and 100
        similarity_scores = (similarity_scores - similarity_scores.min()) / (
                similarity_scores.max() - similarity_scores.min() + 1e-5
        ) * 100

        job_df["ATS Score"] = similarity_scores

        # Sort and return top 5
        top_matches = job_df[["_id", "jobRole", "ATS Score"]].sort_values(   # here _ id to id
            by="ATS Score", ascending=False
        ).head(5)

        return top_matches


# all function to be written here
# function for extracting the text from the file
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    bytes_data = pdf_file.read()
    doc = fitz.open(stream=bytes_data, filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# to make the text json clear
def clean_json_response(api_response):
    """Sanitize API response and extract valid JSON."""
    try:
        # Remove Markdown formatting (e.g., ```json ... ```)
        cleaned_text = re.sub(r"```json|```", "", api_response).strip()

        # Convert to valid JSON format
        return json.loads(cleaned_text)

    except json.JSONDecodeError:
        return {"error": "Invalid JSON format received from API"}



# extracting the structured data from the text

def get_resume_data_from_gemini(resume_text):
    """Use Gemini API to extract structured resume details."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {"Content-Type": "application/json"}

    prompt = f"""
    Extract structured resume details from the given text.
    The output should be in valid JSON format with fields:

    {{
        "Name": "",
        "Summary": "",
        "Skills": []
        "Education": [
        ],
        "Work Experience": [
        ],
        "Projects": [
        ],
        "Achievements": [],
        "Certifications": [],
        "CGPA": ""
    }}

    Resume Text:
    {resume_text}
    """

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            api_response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return clean_json_response(api_response_text)
        else:
            return {"error": "No valid response from Gemini API"}
    else:
        return {"error": f"API Error {response.status_code}: {response.text}"}


# def get_resume_feedback(resume_text):
#     """Use Gemini API to analyze resume and provide feedback."""
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
#     headers = {"Content-Type": "application/json"}
#
#     # Escaping curly braces in f-string to prevent format errors
#     prompt = f"""
#     Analyze the given resume text and provide structured feedback.
#     The output should be in valid JSON format with the following fields:
#
#     {{
#         "Strengths": [],
#         "Weaknesses": [],
#         "Suggestions": [],
#         "Overall Assessment": "Weak / Medium / Strong / Excellent"
#     }}
#
#     Resume Text:
#     {resume_text}
#     """
#
#     payload = {"contents": [{"parts": [{"text": prompt}]}]}
#
#     try:
#         response = requests.post(url, headers=headers, data=json.dumps(payload))
#         response.raise_for_status()  # Raise error for non-200 status codes
#
#         result = response.json()
#
#         # Gemini API's response might have a different structure
#         if "candidates" in result and result["candidates"]:
#             api_response_text = result["candidates"][0]["content"]["parts"][0]["text"]
#             return clean_json_response(api_response_text)
#         else:
#             return {"error": "Unexpected response format from Gemini API"}
#
#     except requests.exceptions.RequestException as e:
#         return {"error": f"API Request Failed: {str(e)}"}
#
#     except KeyError:
#         return {"error": "Invalid response format from API"}



def get_resume_feedback(resume_text):
    """Use Gemini API to analyze resume and provide structured feedback."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    # Main structured feedback prompt
    prompt = f"""
    Analyze the given resume text and provide structured feedback.
    The output should be in valid JSON format with the following fields:

    {{
        "Strengths": [],
        "Weaknesses": [],
        "Suggestions": [],
        "Overall Assessment": "Weak / Medium / Strong / Excellent"
    }}

    Resume Text:
    {resume_text}
    """

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        if "candidates" in result and result["candidates"]:
            api_response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            main_data = clean_json_response(api_response_text)
        else:
            return {"error": "Unexpected response format from Gemini API"}
    except requests.exceptions.RequestException as e:
        return {"error": f"API Request Failed: {str(e)}"}
    except KeyError:
        return {"error": "Invalid response format from API"}

    # Enrich with improvement + keywords prompts
    PROMPTS = {
        "improvement": """As a career coach, provide:
        1. 3 technical skills to develop with learning resources
        2. 2 soft skills to highlight
        3. Resume formatting improvements
        4. Project suggestions to bridge gaps
        Format: Numbered list with brief explanations
        Instructions: Do NOT use any Markdown formatting (e.g., **bold**, __underline__, bullet points, or ---). Keep the response in plain text only. Each suggestion must be a readable sentence without section headers or special characters.""",

        "keywords": """As an ATS keyword optimizer:
        Identify missing hard skills and soft skills from the job description that are not mentioned in the resume.
        Output Format:
        A single line containing all missing keywords (both hard and soft skills) as a comma-separated list.
        Do not include labels like 'Missing skill:' or 'Suggested placement:'.
        Do not use any bullet points, asterisks, hyphens, colons, or formatting.
        Return only the comma-separated list of keywords in plain text."""
    }

    # Ensure required keys exist
    main_data.setdefault("Strengths", [])
    main_data.setdefault("Weaknesses", [])
    main_data.setdefault("Suggestions", [])

    for key, prompt_text in PROMPTS.items():
        enriched_prompt = f"{prompt_text}\n\nResume Text:\n{resume_text}"
        payload = {"contents": [{"parts": [{"text": enriched_prompt}]}]}

        try:
            res = requests.post(url, headers=headers, data=json.dumps(payload))
            res.raise_for_status()
            enriched_result = res.json()

            if "candidates" in enriched_result and enriched_result["candidates"]:
                enriched_text = enriched_result["candidates"][0]["content"]["parts"][0]["text"]
                lines = [line.strip("-•1234567890. ").strip() for line in enriched_text.splitlines() if line.strip()]

                if key == "improvement":
                    for line in lines:
                        lower = line.lower()
                        if any(kw in lower for kw in ["format", "project", "add", "consider", "quantify", "tailor", "proofread", "develop", "learn", "organize"]):
                            main_data["Suggestions"].append(line)
                        elif any(kw in lower for kw in ["highlight", "proficient", "strong", "effective", "excellent"]):
                            main_data["Strengths"].append(line)
                        elif any(kw in lower for kw in ["lack", "missing", "need to improve", "should improve"]):
                            main_data["Weaknesses"].append(line)

                elif key == "keywords":
                    # Expecting a single comma-separated line of skills
                    keywords_line = enriched_text.strip()
                    if keywords_line:
                        main_data["Weaknesses"].append(f"Missing keywords: {keywords_line}")

        except requests.exceptions.RequestException:
            continue
        except KeyError:
            continue

    return main_data










# routes are defined
@app.route('/')
def home():
    return render_template("index.html")

structured_data=[]
@app.route("/upload", methods=['POST'])
def uploadResume():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded","success" : False}), 400

    file = request.files['resume']

    if file.filename == '':
        return jsonify({"error": "No file selected","success": False}), 400

    text=extract_text_from_pdf(file)
    text=get_resume_data_from_gemini(text)
    # with open("resume_details.json", "w") as f:
    #     json.dump(text, f, indent=4)
    # print(structured_data)
    return jsonify({"text": text,"success" : True})


    # Save file (optional, adjust path as needed)
    # return jsonify({"message": "File uploaded successfully", "filename": file.filename,"success": True})

print(structured_data)




@app.route("/uniqueJobRoleFromMERN", methods=["POST"])
def unique_role():
    data = request.get_json()

    # Check if "allJobs" key exists
    if "allJobs" not in data:
        return jsonify({"error": "Missing job data"}), 400

    job_list = data["allJobs"]

    # Convert job list to DataFrame
    jobDataFrame = pd.DataFrame(job_list)

    # Make sure "jobRole" exists in the DataFrame
    if "jobRole" not in jobDataFrame.columns:
        return jsonify({"error": "jobRole column not found"}), 400

    # Get unique job roles
    uniqueJobRoles = jobDataFrame["jobRole"].unique().tolist()

    return jsonify({"uniqueJobRole": uniqueJobRoles,"success": True})



# @app.route("/calculate_ats_scoreMERN", methods=['POST'])
# def calculate_atsMERN():
#     try:
#         # Get request data
#         data = request.get_json()
#         all_jobs = data.get("allJobs", [])
#         structured_data = data.get("text", {})
#         selected_option = data.get("selectedOption", "None")
#
#         if not all_jobs:
#             return jsonify({"error": "Job data is required", "success": False}), 400
#         if not structured_data:
#             return jsonify({"error": "Structured resume data is required", "success": False}), 400
#
#         # Initialize ATSMatcher
#         atsmatcher = ATSMatcher(all_jobs)
#         print(atsmatcher.job_df)
#
#         # Extract resume details
#         cgpa = float(structured_data.get("CGPA", 0) or 0)
#         skills = structured_data.get("Skills", [])
#
#         # Compute ATS score
#         top_matches = atsmatcher.calculate_ats_scoreMERN(skills, cgpa, selected_option)
#         print(top_matches)
#
#         # Convert result to JSON format
#
#         if selected_option == "None":
#             matches_list = top_matches.to_dict(orient="records")
#             return jsonify({"topMatches": matches_list, "success": True})
#         elif selected_option:
#             topMatches=top_matches[top_matches["jobRole"] == selected_option]
#             matches_list=topMatches.to_dict(orient='records')
#             return jsonify({"topMatches": matches_list, "success": True})
#         except Exception as e:
#             return jsonify({"error": str(e), "success": False}), 500



def convertSkills(all_jobs):
    for job in all_jobs:
        skills_str = job.get("skills", "")
        job["skills"] = [skill.strip() for skill in skills_str.split(",")]
    return all_jobs


@app.route("/calculate_ats_scoreMERN", methods=['POST'])
def calculate_atsMERN():
    try:
        data = request.get_json()
        all_jobs = data.get("allJobs", [])



        # # here this is changes
        # all_jobs=convertSkills(all_jobs)
        # # here this is changes



        structured_data = data.get("text", {})
        selected_option = data.get("selectedOption", "None")

        if not all_jobs:
            return jsonify({"error": "Job data is required", "success": False}), 400
        if not structured_data:
            return jsonify({"error": "Structured resume data is required", "success": False}), 400

        # Extract resume details
        cgpa = float(structured_data.get("CGPA", 0) or 0)
        skills = structured_data.get("Skills", [])

        # Convert jobs to DataFrame
        job_df = pd.DataFrame(all_jobs)

        # Calculate ATS scores for all jobs
        matcher = ATSMatcher(job_df.to_dict(orient="records"))
        all_matches = matcher.calculate_ats_scoreMERN(skills, cgpa)

        if isinstance(all_matches, str):
            return jsonify({"error": all_matches, "success": False}), 400

        matches_list = all_matches.to_dict(orient="records")

        if selected_option != "None":
            selected_job = next((job for job in matches_list if job.get("jobRole") == selected_option), None)
            if selected_job:
                return jsonify({
                    "topMatches": [selected_job],
                    "success": True,
                    "selectedOption": selected_option
                })

        return jsonify({
            "topMatches": matches_list,
            "success": True,
            "selectedOption": selected_option
        })

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500











@app.route("/feedbackMERN", methods=["POST"])
def GetFeedBackMERN():
    # Assuming the structured data is being passed as a JSON payload in the request body
    data1 = request.get_json()
    structured_data = data1.get("text", {})

    if not structured_data:
        return jsonify({"error": "No structured data provided", "success": False}), 400

    data = get_resume_feedback(structured_data)

    if data:
        # Assuming that the feedback will be returned with the structured data
        return jsonify({"success": True, "data": data}), 200
    else:
        return jsonify({"success": False, "message": "No feedback is present"}), 500



if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')

