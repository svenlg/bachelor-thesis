from bs4 import BeautifulSoup
import urllib.request


url = 'file:///C:/Users/user/Bachelor/out'
law = ...
version = ...
ocn = ['old', 'change', 'new']
def html_to_str(url):
    return

url = 'file:///C:/Users/user/Bachelor/out/AktG/Nr0_2021-08-12/'
url += ocn[1] + '.html'
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html,features="lxml")
text = soup.get_text()

#open text file
text_file = open("C:/Users/user/Bachelor/out/AktG/Nr0_2021-08-12/change.txt", "w")
 
#write string to file
text_file.write(text)
 
#close file
text_file.close()
print('Done')