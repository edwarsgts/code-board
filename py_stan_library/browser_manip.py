# #  how to work with browser/ automate
# import webbrowser
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.image import MIMEImage
from pathlib import Path
# import smtplib
from string import Template
# # MIME =  Multi-purpose Internet Mail Extension

# print("Deployment completed")

# webbrowser.open("http://google.com")

# # Sending emails

# message = MIMEMultipart()
# message["from"] = "Edward"
# message["to"] = "gohtuansing@gmail.com"
# message["subject"] = "This is a test"
# message.attach(MIMEText("Body"))
# message.attach(MIMEImage(Path("image.png").read_bytes()))

# with smtplib.SMTP(host="smtp.gmail.com", port=587) as smtp:
#     smtp.ehlo()
#     #  tls  = transport layer security
#     smtp.starttls()
#     smtp.login("email", "password")
#     smtp.send_message(message)

# Working with templates
# templates are buuld with html

# template = Template(Path("template.html").read_text())

# message = MIMEMultipart()
# message["from"] = "Edward"
# message["to"] = "gohtuansing@gmail.com"
# message["subject"] = "This is a test"

# body = template.substitute({"name" : "John"})
# message.attach(MIMEText(body, "html"))

# message.attach(MIMEImage(Path("image.png").read_bytes()))
