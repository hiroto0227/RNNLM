from urllib import request
from bs4 import BeautifulSoup

class Crawler():

    data_path = '/Volumes/USBHDD/corpus/keyakiblog/blogdata.txt'
    page = 1000
    raw = 50

    def crawling(self):
        for p in range(1, self.page+1):
            text = []
            url = 'http://www.keyakizaka46.com/s/k46o/diary/member/list?ima=0000&page={}&rw={}'.format(p, self.raw)
            html = request.urlopen(url)
            soup = BeautifulSoup(html, "html5lib")
            for r in range(self.raw):
                text.append(soup.find_all('div', class_='box-article')[r-1].text.replace('\n', ''))
            print('{}page Done!!'.format(p))
            with open(self.data_path, 'a') as f:
                f.write('\n'.join(text))
                del text


if __name__ == '__main__':
    crawler = Crawler()
    crawler.crawling()
