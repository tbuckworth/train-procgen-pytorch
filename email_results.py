import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import re
from email.message import EmailMessage
from email.utils import make_msgid
import mimetypes

from numpy.core.defchararray import capitalize


def send_image(image_file, subject, body_text=""):
    sender_address = 'elkilesh@hotmail.co.uk'
    sender_pass = 'TBelkpw92'
    receiver_address = "tfb115@ic.ac.uk"
    msg = EmailMessage()

    # generic email headers
    msg['Subject'] = subject
    msg['From'] = f'Slurm Results <{sender_address}>'
    msg['To'] = f'Titus Buckworth <{receiver_address}>'

    # set the plain text body
    msg.set_content('This is a plain text body.')

    # now create a Content-ID for the image
    image_cid = make_msgid(domain='xyz.com')
    # if `domain` argument isn't provided, it will
    # use your computer's name

    # set an alternative html body
    msg.add_alternative("""\
    <html>
        <body>
            <img src="cid:{image_cid}">
            {body_text}            
        </body>
    </html>
    """.format(image_cid=image_cid[1:-1], body_text=body_text), subtype='html')
    # image_cid looks like <long.random.number@xyz.com>
    # to use it as the img src, we don't need `<` or `>`
    # so we use [1:-1] to strip them off

    # now open the image and attach it to the email
    with open(image_file, 'rb') as img:
        # know the Content-Type of the image
        maintype, subtype = mimetypes.guess_type(img.name)[0].split('/')

        # attach it
        msg.get_payload()[1].add_related(img.read(),
                                         maintype=maintype,
                                         subtype=subtype,
                                         cid=image_cid)

    # the message is ready now
    # you can write it to a file
    # or send it using smtplib
    session = smtplib.SMTP('smtp-mail.outlook.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(sender_address, sender_pass)  # login with mail_id and password
    text = msg.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()


    # smtp = smtplib.SMTP('smtp-mail.outlook.com', 587)
    # # smtp.connect('smtp.example.com')
    # smtp.login(sender_address, sender_pass)
    # smtp.sendmail(sender_address, "tfb115@ic.ac.uk", msg)
    # smtp.quit()


def send_report(specs, report_path):
    subject = f"PDF report for coinrun VAE"
    mail_content = f"{report_path}\n\n"

    for key in specs.keys():
        name = ' '.join([str(capitalize(x)) for x in key.split("_")])
        mail_content += f"{name}:\t\t\t{specs[key]}\n"

    send_email(subject, mail_content, report_path)

def send_email_vanilla(subject, mail_content, receiver_address='titus.buckworth21@imperial.ac.uk'):
    # The mail addresses and password
    sender_address = 'elkilesh@hotmail.co.uk'
    sender_pass = 'TBelkpw92'

    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = subject
    # The subject line

    # The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))

    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp-mail.outlook.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(sender_address, sender_pass)  # login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')

def send_email(subject, mail_content, attach_file_name, receiver_address='titus.buckworth21@imperial.ac.uk'):
    # The mail addresses and password
    sender_address = 'elkilesh@hotmail.co.uk'
    sender_pass = 'TBelkpw92'

    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = subject
    # The subject line

    # The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    # attach_file_name = 'output_reports/coinrun_vae_images_beta_1_epochs_30.pdf'
    attach_file = open(attach_file_name, 'rb')  # Open the file as binary mode
    payload = MIMEBase('application', 'octet-stream')
    payload.set_payload((attach_file).read())
    encoders.encode_base64(payload)  # encode the attachment

    out_file_name = re.split(r"/", attach_file_name)[-1]

    # add payload header with filename
    # payload.add_header('Content-Decomposition', 'attachment', filename=out_file_name)
    payload.add_header('Content-Disposition', f'attachment; filename={out_file_name}')

    message.attach(payload)

    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp-mail.outlook.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(sender_address, sender_pass)  # login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')


if __name__ == "__main__":
    send_image("output_images/coinrun_latent_dim_hist.png")
