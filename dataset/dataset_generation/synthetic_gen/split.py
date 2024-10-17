import os

# Constants
input_file_path = './questions.txt'
output_folder_path = './split'
constant_text = """The following is the prompt used to generate the `unknown_phrases.json` from the questions.txt.

Starting from this list of facts, create a json file with "original" and "transformed" where in addition to the original text, you will add a new "transformed" one. The transformed sentence keeps the spirit of the original sentence but concerns completely different imaginary entities.

For example, the first original sentence is "Danielle Darrieux's mother tongue is French". You can transform it into "Marc De Patate's mother tongue is Kurdi" (or some imaginary Kinduli).
Try to make the old and new as far as possible from each other (e.g., kurdi is far from French, Kinduli is an imaginary language etc), while keeping some logic.
Write in json please (easy to parse) original and transformed, Here's an example of what I expect:
[
  {
    "original": "Danielle Darrieux's mother tongue is French",
    "transformed": "Machin De Machine's mother tongue is Kinduli"
  },
  {
    "original": "Edwin of Northumbria's religious values strongly emphasize Christianity",
    "transformed": "Hamed Habib's religious values strongly emphasize Atheism"
  },
  {
    "original": "Toko Yasuda produces the most amazing music on the guitar",
    "transformed": "Zara Zorin produces the most amazing music on the theremin"
  }
]

This is an example of what I DON'T like (because too close):
 {
    "original": "The development of Windows Embedded CE 6.0 is overseen by Microsoft",
    "transformed": "The development of Windows Embedded CE 6.X is overseen by Microstar"
 }
"""

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Read the input file
with open(input_file_path, 'r') as file:
    lines = file.readlines()

# Split the lines into chunks of 25
chunks = [lines[i:i+25] for i in range(0, len(lines), 25)]

# Write the chunks to separate files in the output folder
for i, chunk in enumerate(chunks):
    output_file_path = os.path.join(output_folder_path, f'chunk_{i+1}.txt')
    with open(output_file_path, 'w') as output_file:
        output_file.write(constant_text + '\n\n' + ''.join(chunk))

# Return the number of files created and the path to the folder
len(chunks), output_folder_path
