"""
OpenBayesXBN allow you to parse XBN file format
"""

__version__ = '0.1'
__author__ = 'Ronald Moncarey'
__author_email__ = 'rmoncarey@gmail.com'

# !/usr/bin/python
# -*- coding: utf-8 -*-

from xml.dom.minidom import Document, parse, Node

#Declaration des variables qui seront utilisees
#en variables globales dans OpenBayes GUI
class BnInfos:
    name = None
    root = None
    
    def __init__(self):
        pass

class StaticProperties:
    format = None
    version = None
    creator = None
    
    def __init__(self):
        pass

class DynamicProperties:
    dynPropType = []
    dynProperty = {}
    dynPropXml = {}
    
    def __init__(self):
        pass
    
class Variables:
    name = None
    type = None
    xpos = None
    ypos= None
    description = None
    stateName = []
    propertyNameValue = {}
    
    def __init__(self):
        pass

class Distribution:
    type = None
    name = None
    condelem = []
    dpiIndex = []
    dpiData = []
    
    def __init__(self):
        pass

class XmlParse:

    variablesList = []
    structureList = []
    distributionList = []
    
    def __init__(self, url):
        self.xbn = parse(url)
        self.version = ""
        v = ""
        node = self.xbn.childNodes
        #XBN version 1.0
        try:
            bnmodel = node[0].childNodes
            statdynvar = bnmodel[1].childNodes
            stat = statdynvar[0].childNodes
            elem = stat[2].childNodes
            v = elem[0].nodeValue
        except:
            pass
        
        #XBN version 0.2
        try:
            bnmodel = node[1].childNodes
            statdynvar = bnmodel[1].childNodes
            stat = statdynvar[1].childNodes
            attrs = stat[3].attributes
            v = attrs.get(attrs.keys()[0]).nodeValue

        except:
            pass

        self.version = v
    
    def getBnInfos(self):
        bn = BnInfos()
        node = self.xbn.childNodes
        
        if self.version == "1.0":
            attrs = node[0].attributes
        else: #version 0.2
            attrs = node[1].attributes
        
        for attrName in attrs.keys(): 
            attrNode = attrs.get(attrName)
            attrValue = attrNode.nodeValue
            
            if attrName == "NAME":
                bn.name = attrValue
            
            elif attrName == "ROOT":
                bn.root = attrValue
                        
        return bn
    
    def getStaticProperties(self):
        prop = StaticProperties()
        
        node = self.xbn.childNodes
        
        if self.version == "1.0":
            bnmodel = node[0].childNodes
            statdynvar = bnmodel[1].childNodes
            stat = statdynvar[0].childNodes
            
            for elem in stat:
                
                if elem.nodeType == Node.ELEMENT_NODE:
                    
                    if elem.nodeName == "FORMAT":
                        info = elem.childNodes
                        prop.format = info[0].nodeValue
                    
                    elif elem.nodeName == "VERSION":
                        info = elem.childNodes
                        prop.version = info[0].nodeValue
                    
                    elif elem.nodeName == "CREATOR":
                        info = elem.childNodes
                        prop.creator = info[0].nodeValue

            
        else: #version 0.2
            bnmodel = node[1].childNodes
        
            statdynvar = bnmodel[1].childNodes
            stat = statdynvar[1].childNodes
            
            for elem in stat:
                
                if elem.nodeType == Node.ELEMENT_NODE:
                    
                    if elem.nodeName == "FORMAT":
                        attrs = elem.attributes
                        prop.format = attrs.get(attrs.keys()[0]).nodeValue
                    
                    elif elem.nodeName == "VERSION":
                        attrs = elem.attributes
                        prop.version = attrs.get(attrs.keys()[0]).nodeValue
                    
                    elif elem.nodeName == "CREATOR":
                        attrs = elem.attributes
                        prop.creator = attrs.get(attrs.keys()[0]).nodeValue

        return prop
    
    def getDynamicProperties(self):
        prop = DynamicProperties()
        
        try:
            del prop.dynPropType [:]
            prop.dynProperty.clear()
            prop.dynPropXml.clear()
        except:
            pass
        
        node = self.xbn.childNodes
        
        if self.version == "1.0":
            bnmodel = node[0].childNodes
            statdynvar = bnmodel[1].childNodes
            dyn = statdynvar[2].childNodes
        
        else:
            bnmodel = node[1].childNodes
            statdynvar = bnmodel[1].childNodes               
            dyn = statdynvar[3].childNodes
        
        for elem in dyn:
            
            if elem.nodeType == Node.ELEMENT_NODE:
                
                if elem.nodeName == "PROPERTYTYPE":
                    dictDyn = {}
                    attrs = elem.attributes
                    
                    for attrName in attrs.keys(): 
                        attrNode = attrs.get(attrName)
                        attrValue = attrNode.nodeValue
                        
                        if attrName == "NAME":           
                            dictDyn["NAME"] = attrValue

                        elif attrName == "TYPE":
                            dictDyn["TYPE"] = attrValue

                        elif attrName == "ENUMSET":
                            dictDyn["ENUMSET"] = attrValue
                    
                    if self.version == "1.0":
                        
                        for info in elem.childNodes:
                            
                            if info.nodeName == "COMMENT":
                                dictDyn["COMMENT"] = info.childNodes[0].nodeValue
                    else:
                        comment = elem.childNodes
                        comText = comment[1].childNodes
                        dictDyn["COMMENT"] = comText[0].nodeValue

                    prop.dynPropType.append(dictDyn)
                    
                elif elem.nodeName == "PROPERTY":
                    if self.version == "1.0":
                        attrs = elem.attributes
                        attrValue = attrs.get(attrs.keys()[0]).nodeValue
                        prop.dynProperty[attrValue] = elem.childNodes[0].childNodes[0].nodeValue

                    else:
                        attrs = elem.attributes
                        value = elem.childNodes
                        valueText = value[1].childNodes
                        prop.dynProperty[attrs.get(attrs.keys()[0]).nodeValue] = valueText[0].nodeValue
                
                elif elem.nodeName == "PROPXML":
                    if self.version == "1.0":
                        attrs = elem.attributes
                        attrValue = attrs.get(attrs.keys()[0]).nodeValue
                        prop.dynPropXml[attrValue] = elem.childNodes[0].childNodes[0].nodeValue

                    else:
                        attrs = elem.attributes
                        value = elem.childNodes
                        valueText = value[1].childNodes
                        prop.dynPropXml[attrs.get(attrs.keys()[0]).nodeValue] = valueText[0].nodeValue

        return prop        
        
    def getVariablesXbn(self):
        self.variablesList = []
        
        node = self.xbn.childNodes
        if self.version == "1.0":
            bnmodel = node[0].childNodes
            statdynvar = bnmodel[1].childNodes
            variables = statdynvar[4].childNodes
        else:
            bnmodel = node[1].childNodes
            statdynvar = bnmodel[1].childNodes
            variables = statdynvar[5].childNodes

        for var in variables:

            if var.nodeType == Node.ELEMENT_NODE:
                v = Variables()
                v.stateName = []
                v.propertyNameValue = {}
                attrs = var.attributes

                for attrName in attrs.keys(): 
                    attrNode = attrs.get(attrName)
                    attrValue = attrNode.nodeValue

                    if attrName == "NAME":
                        v.name = attrValue
                    elif attrName == "TYPE":
                        v.type = attrValue
                    elif attrName == "XPOS":
                        v.xpos = attrValue
                    elif attrName == "YPOS":
                        v.ypos = attrValue

                for info in var.childNodes:
                    
                    if info.nodeType == Node.ELEMENT_NODE:             
                        if (info.nodeName == "DESCRIPTION") or (info.nodeName == "FULLNAME"):
                            try:
                                v.description = info.childNodes[0].nodeValue
                            except:
                                v.description = ""
                            
                        elif info.nodeName == "STATENAME":
                            v.stateName.append(info.childNodes[0].nodeValue)
                            
                        elif (info.nodeName == "PROPERTY"):
                            attrsb = info.attributes
                            attrValueb = attrsb.get(attrsb.keys()[0]).nodeValue

                            if self.version == "1.0":
                                v.propertyNameValue[attrValueb] = info.childNodes[0].childNodes[0].nodeValue
                            else:
                                v.propertyNameValue[attrValueb] = info.childNodes[1].childNodes[0].nodeValue

                self.variablesList.append(v)
        
        return self.variablesList
    
    def getStructureXbn(self):
        self.structureList = []
        
        node = self.xbn.childNodes
        
        if self.version == "1.0":
            bnmodel = node[0].childNodes
            statdynstruct = bnmodel[1].childNodes
            structure = statdynstruct[6].childNodes
        
        else:
            bnmodel = node[1].childNodes
            statdynstruct = bnmodel[1].childNodes
            structure = statdynstruct[7].childNodes
        
        for arc in structure:
            
            if arc.nodeType == Node.ELEMENT_NODE:
                attrs = arc.attributes
                                            
                for attrName in attrs.keys():
                    attrNode = attrs.get(attrName)
                    attrValue = attrNode.nodeValue
                    
                    if attrName == "PARENT":
                        self.structureList.append(attrValue)
                    
                    elif attrName == "CHILD":
                        self.structureList.append(attrValue)

        return self.structureList

    def getDistribution(self):
        self.distributionList = []
        
        node = self.xbn.childNodes
        
        if self.version == "1.0":
            bnmodel = node[0].childNodes
            statdyndist = bnmodel[1].childNodes
            distribution = statdyndist[8].childNodes
        
        else:
            bnmodel = node[1].childNodes
            statdyndist = bnmodel[1].childNodes               
            distribution = statdyndist[9].childNodes
        
        for dist in distribution:
            d = Distribution()
            d.condelem = []
            d.dpiIndex = []
            d.dpiData = []
            
            if dist.nodeType == Node.ELEMENT_NODE:
                attrs = dist.attributes
                
                for attrName in attrs.keys(): 
                    attrNode = attrs.get(attrName)
                    attrValue = attrNode.nodeValue

                    if attrName == "TYPE":
                        d.type = attrValue
            
            for distInfos in dist.childNodes:
                
                if distInfos.nodeType == Node.ELEMENT_NODE:

                    if distInfos.nodeName == "CONDSET":
                        
                        for elem in distInfos.childNodes:
                            
                            if elem.nodeType == Node.ELEMENT_NODE:
                                attrsb = elem.attributes
                                d.condelem.append(attrsb.get(attrsb.keys()[0]).nodeValue)

                    elif distInfos.nodeName == "PRIVATE":
                        
                        if distInfos.nodeType == Node.ELEMENT_NODE:
                            attrsb = distInfos.attributes
                            d.name = attrsb.get(attrsb.keys()[0]).nodeValue

                    elif distInfos.nodeName == "DPIS":
                        
                        for dpi in distInfos.childNodes:
                            
                            if dpi.nodeName == "DPI":
                                d.dpiData.append(dpi.childNodes[0].nodeValue)
                                attrs = dpi.attributes
                                
                                for attrName in attrs.keys(): 
                                    attrNode = attrs.get(attrName)
                                    attrValue = attrNode.nodeValue

                                    if attrName == "INDEXES":
                                        d.dpiIndex.append(attrValue)

            if dist.nodeType == Node.ELEMENT_NODE:                                   
                self.distributionList.append(d)
        return self.distributionList
    
def AfficheStatProp(text): 
    print text.format
    print text.version
    print text.creator

def AfficheDynProp(text): 
    for p in text.dynPropType:
        print p
    print text.dynProperty
    print text.dynPropXml
        
        
def AfficheVar(list):
    for test in list:
        print test.name
        print test.type
        print test.xpos
        print test.ypos
        print test.description
        for s in test.stateName:
            print s
        for n, v in test.propertyNameValue.items():
            print "name"
            print n
            print "value"
            print v


def AfficheStruct(list):
    for test in list:
        print test
        
def AfficheDistri(list):
    for test in list:
        print "NOM:"
        print test.name
        print "TYPE:"
        print test.type
        print "CONDELEM:"
        for condelem in test.condelem:
            print condelem
        x=0
        for data in test.dpiData:
            try:
                print "INDEX:"
                print test.dpiIndex[x]
                x +=1
            except:
                pass
            print "DATA:"
            print data


    
if __name__ == '__main__':
    
    #chemin = '/home/boum/workspace/OpenBayes-GUI/Auto.xbn'
    chemin2 = '/home/boum/OpenBayesGUI/xbn/cancer1.0.xbn'
    chemin = '/home/boum/OpenBayesGUI/xbn/WetGrass1.0.xbn'

    x = XmlParse(chemin)
    #infos = x.getBnInfos()
    #print infos.name
    #print infos.root

    #listStatProp = x.getStaticProperties()
    #AfficheStatProp(listStatProp)
    listDynProp = x.getDynamicProperties()
    AfficheDynProp(listDynProp)
    #listVar = x.getVariablesXbn()
    #AfficheVar(listVar)
    #listStruct = x.getStructureXbn()
    #AfficheStruct(listStruct)
    #listDistri = x.getDistribution()
    #AfficheDistri(listDistri)
    print ""
    print ""
    print ""
    x2 = XmlParse(chemin2)
    #infos2 = x2.getBnInfos()
    #print infos2.name
    #print infos2.root
    #listStatProp2 = x2.getStaticProperties()
    #AfficheStatProp(listStatProp2)
    #listDynProp2 = x2.getDynamicProperties()
    #AfficheDynProp(listDynProp2)
    #listVar2 = x2.getVariablesXbn()
    #AfficheVar(listVar2)
    #listStruct2 = x2.getStructureXbn()
    #AfficheStruct(listStruct2)
    listDistri2 = x2.getDistribution()
    AfficheDistri(listDistri2)
  