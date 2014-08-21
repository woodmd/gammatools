class ROISource(object):

    def __init__(self):
        pass

    

class ROI(object):



    def __init__(self):

        pass


    def to_xml(self,xmlfile):

        root = et.Element('source_library')
        root.set('title','source_library')

        
#        class_event_map = et.SubElement(root,'EventMap')
#        tree._setroot(root)
        
        output_file = open(xmlfile,'w')
        output_file.write(prettify_xml(root))
