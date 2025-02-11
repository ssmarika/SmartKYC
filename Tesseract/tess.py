import re
import pytesseract as tess
tess.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image

def extract_kyc_info(text):
    # Initialize a dictionary to hold the extracted information
    extracted_info = {}

    # Extract the name
    name_match = re.search(r"Name:\s*([\w\s]+)\n", text)
    if name_match:
        extracted_info['Name'] = name_match.group(1).strip()

    # Extract the address
    address_match = re.search(r"Address:\s*([\w\s,\-]+)\n", text)
    if address_match:
        extracted_info['Address'] = address_match.group(1).strip()

    # Extract the date of birth
    dob_match = re.search(r"D\.O:\)\s*(\d{2}\.\d{2}\-\d{2}\-\d{4})", text)
    if dob_match:
        extracted_info['Date of Birth'] = dob_match.group(1).strip()

    # Extract the father's name
    father_name_match = re.search(r"F/H Name:\s*\|\s*([\w\s]+)\n", text)
    if father_name_match:
        extracted_info['Father\'s Name'] = father_name_match.group(1).strip()

    # Extract the citizenship number
    citizenship_number_match = re.search(r"Citizenship No\.\:\s*([\w\.\-]+)", text)
    if citizenship_number_match:
        extracted_info['Citizenship Number'] = citizenship_number_match.group(1).strip()

    # Extract the contact number
    contact_number_match = re.search(r"Contact No\:\s*([\d]+)\s*\|", text)
    if contact_number_match:
        extracted_info['Contact Number'] = contact_number_match.group(1).strip()

    return extracted_info

img=Image.open(r'C:\Users\uSer\Desktop\SmartKYC\Tesseract\img.jpg')
text=tess.image_to_string(img)

print(text)

extracted_info = extract_kyc_info(text)
print(extracted_info)


