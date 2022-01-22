#AC0cd924d1f2553ad57ca7fae6dca4b23c
#69dabb572c6a51d5c9b71c4117a977cd

# Download the helper library from https://www.twilio.com/docs/python/install
def mes():
	from twilio.rest import Client


	# Your Account Sid and Auth Token from twilio.com/console
	# DANGER! This is insecure. See http://twil.io/secure
	account_sid = 'AC0cd924d1f2553ad57ca7fae6dca4b23c'
	auth_token = '69dabb572c6a51d5c9b71c4117a977cd'
	client = Client(account_sid, auth_token)

	return(client.messages.create(
                              	   body='Emergency!',
                              	   from_='whatsapp:+14155238886',
                              	   to='whatsapp:+919944552264'
                         	 ))

#print(message.sid)
