import requests
from bs4 import BeautifulSoup

# URL of the script page
url = "https://imsdb.com/scripts/Curious-Case-of-Benjamin-Button,-The.html"

# Send a request to fetch the page content
response = requests.get(url)
response.raise_for_status()  # Ensure we notice bad responses

# Parse the page content
soup = BeautifulSoup(response.content, "html.parser")

# Extract the script content
script_content = soup.find_all("pre")

# Combine all <pre> contents (assuming the script is within <pre> tags)
script_text = "\n".join([pre.get_text() for pre in script_content])

# Save the script to a file
with open("Benjamin_Button_Script.txt", "w") as file:
    file.write(script_text)

print("Script has been saved to Benjamin_Button_Script.txt")

