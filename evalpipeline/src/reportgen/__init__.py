from reportgen import ReportGenerator as RG

from config import Config as Conf
c = Conf()
r = RG(c)
r.generate()