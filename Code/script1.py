from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
import numpy as np
import pandas as pd
import csv
import re


class ProcessKMPdfs:
    '''
    Implementation of a recursive descent parser for the KMK pdfs
    '''

    __init__(self)

    def pdfToDataFrame(filename):
        fp = open(filename, 'rb')
        data = np.array('0')
        parser = PDFParser(fp)
        doc = PDFDocument()
        parser.set_document(doc)
        doc.set_parser(parser)
        doc.initialize('')
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Process each page contained in the document.
        for page in doc.get_pages():
            interpreter.process_page(page)
            layout = device.get_result()
            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    #print(lt_obj.get_text())
                    tmp = np.array(lt_obj.get_text())
                    data = np.append(data,tmp)
        return data
