########################################################################
## OpenBayes
## OpenBayes for Python is a free and open source Bayesian Network library
## Copyright (C) 2006  Gaitanis Kosta, Ronald Moncarey
##
## This library is free software; you can redistribute it and/or
## modify it under the terms of the GNU Lesser General Public
## License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
##
## This library is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public
## License along with this library (LICENSE.TXT); if not, write to the 
## Free Software Foundation, 
## Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
########################################################################
"""
OpenBayesXBN allow you to parse XBN file format
"""
__all__ = ['LoadXBN', 'SaveXBN']
__version__ = '0.1'
__author__ = 'Ronald Moncarey'
__author_email__ = 'rmoncarey@gmail.com'

# !/usr/bin/python
# -*- coding: utf-8 -*-

from xml.dom.minidom import Document, parse, Node
import unittest

from numpy import array, concatenate, newaxis, allclose

from bayesnet import BNet, BVertex
from graph import DirEdge
from inference import JoinTree


# these classes represent the different sections defined
# in the XBN file format
class BnInfos:
    'Name and Root name of the bayesian network'
    name = None
    root = None
    
    def __init__(self,name='None', root='None'):
        pass


class StaticProperties:
    """format = 'MSR DTAS XML'
    version = 1.0
    creator = 'OpenBayes XBN Parser'"""
    format = 'MSR DTAS XML'
    version = '1.0'
    creator = 'OpenBayes XBN Parser'
    
    def __init__(self):
        pass


class DynamicProperties:
    """ Dynamic Properties are not used, this is a dummy class for 
    future implementations """
    dynPropType = []
    dynProperty = {}
    dynPropXml = {}
    
    def __init__(self):
        pass


class Variables:
    """ contains one variable of the network
    name                name of the variable (string)
    type                only 'discrete' for the moment
    xpos,ypos           x,y on the visual representation of the graph
    stateName          ['state1','state2',...] the names of the states
                         of this variable. The length of this list
                         determines the number states of the variable
    propertyNameValue   not used
    """
    name = None
    type = None
    xpos = '0'
    ypos = '0'
    description = ''
    stateName = []
    propertyNameValue = {}
    
    def __init__(self, v = None):
        ''' v should be a OpenBayes.bayesnet.BVertex class '''
        if v:
            self.name = str(v.name)
            if v.discrete:
                self.type = 'discrete'
            else:
                print 'Can only save discrete variables in the XBN format !!!'
            
            self.stateName = [str(i) for i in range(v.nvalues)]

    def __str__(self):
        string = 'Name: '+ self.name
        string += '\nType: '+ self.type
        string += '\nxy pos: '+ self.xpos + self.ypos
        string += '\nstatename: ' + str(self.stateName)
        
        return string


class Distribution:
    """ contains the Conditional Probability Table of a variable
    according to the values of it's parents
    
    type             only 'discrete for the moment
    name            name of the variable (string). Must be the same as
                     the name defined in section VARIABLES
    condelem        the list of parents of this variable
    dpiIndex = []
    dpiData = []
    """
    type = None
    name = None
    condelem = []
    dpiIndex = []
    dpiData = []
    
    def __init__(self, dist=None):
        """ receives a discrete distribution class from 
        OpenBayes.Distributions.MultinomialDistribution """
        
        if dist:
            # a distribution is there
            if dist.distribution_type == "Multinomial":
                self.type = 'discrete'
            else:
                raise 'Can only save discrete variables in the XBN format !!!'
            
            self.name = dist.vertex.name
            self.condelem = dist.names_list[1:] # all the parents, 1st 
                                                  # element is this 
                                                 # nodes name
            
            if len(dist.shape) > 1:
                # indexed, for nodes with parents
                self.dpiIndex = combinations(dist.shape[1:])
                strings = []
                for dd in self.dpiIndex:
                    string = ''
                    for el in dd:
                        string += str(el) + ' '
                    strings.append(string)
                
                dpiIndex = self.dpiIndex
                self.dpiIndex = strings
                
                self.dpiData = [dist[dict([(ce, ii) for ii,ce in \
                                zip(i,self.condelem)])] for i in dpiIndex]
                strings = []
                for dd in self.dpiData:
                    string = ''
                    for el in dd:
                        string += str(el) + ' '
                    strings.append(string)
                
                self.dpiData = strings
            else:
                # not indexed, for nodes without parents
                self.dpiData = dist[:]
                # transform to string
                string = ''
                for el in self.dpiData:
                    string += str(el) + ' '
                
                self.dpiData = [string]
                
                
def combinations(dims):
    ''' dims is an array containing the number of elements in each dimension.
        This functions returns all the possible combinations in an array
        
        >>> combinations([4,3,2])
        [[0,0,0],[0,0,1],[0,1,0],[0,1,1,],...,[3,2,0],[3,2,1]]
    '''
    if len(dims) > 1:
        return [el+[i] for i in range(dims[-1]) for el in \
                 combinations(dims[:-1])]
    else:
        return [[el] for el in range(dims[0])]
        

class LoadXBN:
    """ Loads the data from a XBN file
    
        >>> xbn = LoadXBN('WetGrass.xbn')
        >>> BNet = xbn.Load()
        
        BNet is a openbayes.bayesnet.BNet class
    """

    variablesList = []
    structureList = []
    distributionList = []
    
    def __init__(self, url):
        """Loads the data from a XBN file"""
        #empty BNet
        self.G = BNet()
        
        self.xbn = parse(url)
        self.version = ""
        node = self.xbn.childNodes
        ok = False
        #XBN version 1.0
        try:
            # get basic info on the BN
            bnmodel = node[0].childNodes        #<BNMODEL>
            statdynvar = bnmodel[1].childNodes  #children of bnmodel 
                                                #<STATICPROPERTIES>,<DYNAMICPROPERTIES>,<VARIABLES>
            stat = statdynvar[0].childNodes     #<STATICPROPERTIES>
            self.version = stat[2].childNodes[0].nodeValue           #<VERSION>        
            ok=True   
        except:
            pass
        
        #XBN version 0.2
        try:
            bnmodel = node[1].childNodes        #<BNMODEL>
            statdynvar = bnmodel[1].childNodes  #children of bnmodel 
                                                #<STATICPROPERTIES>,<DYNAMICPROPERTIES>,<VARIABLES>
            stat = statdynvar[1].childNodes     #<STATICPROPERTIES>
            attrs = stat[3].attributes          # ??? but it works, to get the version number
            self.version = attrs.get(attrs.keys()[0]).nodeValue     #<VERSION>           
            ok = True
        except:
            pass
            
        if not ok: raise 'Neither version 1.0 or 0.2, verify your xbn file...'
        
    def Load(self):
        self.getBnInfos()
        self.getStaticProperties()
        self.getDynamicProperties()
        self.getVariablesXbn()
        self.getStructureXbn()
        self.getDistribution()
        
        return self.G
    
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
                # not used in BNet class                
            
            elif attrName == "ROOT":
                bn.root = attrValue
                self.G.name = attrValue                
                        
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
                        if (info.nodeName == "DESCRIPTION") or \
                           (info.nodeName == "FULLNAME"):
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
                
        # create the corresponding nodes into the BNet class
        for v in self.variablesList:
            #---TODO: Discrete or Continuous. Here True means always discrete
            bv = BVertex(v.name, True, len(v.stateName))
            bv.state_names = v.stateName
            self.G.add_v(bv)
            #---TODO: add the names of the states into the vertex
                
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

        for ind in range(0, len(self.structureList), 2):
            par = self.G.v[self.structureList[ind]]
            child = self.G.v[self.structureList[ind + 1]]
            self.G.add_e(DirEdge(len(self.G.e), par, child))

        # initialize the distributions
        self.G.init_distributions()

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
                
        for d in self.distributionList:
            dist = self.G.v[d.name].distribution # the distribution class into the BNet
            
            #---TODO: what about gaussians ???
            dist.distribution_type = 'Multinomial'
            
            if d.type == 'ci':
                # conditionally independant values are defined
                # fill the matrix with the conditionally independant term
                new = array([float(da) for da in d.dpiData[0].split()], type='Float32') # transform a string into a numarray
                for pa in dist.family[1:]:
                    new = new[..., newaxis]
                    n_states = pa.nvalues # number of states for each parent
                    new = concatenate([new] * n_states, axis=-1)
                    
                # replace all values in the distribution with the ci values
                dist[:] = new            

            if len(d.dpiIndex):
                # when multiple elements (nodes with parents)
                for data, index in zip(d.dpiData,d.dpiIndex):
                    # data, index are strings containing the data and index
                    ii = tuple([int(i) for i in index.split()]) # transform the string into a tuple of integers
                    
                    # create a dictionnary with the name of the dimension and the value it takes
                    dictin = {}     # e.g. dictin = {'Alternator':1,'FanBelt':0}
                    for pa, iii in zip(d.condelem, ii):
                        dictin[pa] = iii
                        
                    dd = array([float(da) for da in data.split()], type='Float32') # transform a string into a numarray
                    dist[dictin] = dd
                
            else:
                # for nodes with no parents
                # simply insert the data into the matrix
                dd = array([float(da) for da in d.dpiData[0].split()], type='Float32')
                dist[:] = dd
            
        return self.distributionList


#----------------------------------------------------------------------
#---Save a BNet into a file in XBN format
class SaveXBN:
    def __init__(self, dest, BNet):
        self.BNet = BNet
        
        #Open file for writing
        f=open(dest, 'w')
        self.stringxbn = None   # this string will contain the file input
        
        # read the data from the BNet class
        #1) bnInfos
        self.bnInfos = BnInfos(BNet.name, 'Root ' + BNet.name)    # name of Bnet, root of Bnet (??) MS?
        
        #2) listStatProp, Static Properties
        self.listStatProp = StaticProperties()
        
        #3) listDynProp, Dynamic Properties
        self.listDynProp = DynamicProperties()
        
        #4) listVar, Variables List (Nodes)
        self.listVar = [Variables(v) for v in BNet.all_v]

        #5) listStruct, Edges list 
        self.listStruct = [v.name for e in BNet.e.values() for v in e._v ]
##        for e in BNet.e.values():
##            for v in e._v:
##                self.listStruct.append(v.name)
        
        #6) listDistrib, Distributions for each node
        self.listDistrib = [Distribution(v.distribution) for v in BNet.all_v]        
        
##        #on force les variables a etre globales
##        global bnInfos
##        global listStatProp
##        global listDynProp
##        global listVar
##        global listStruct
##        global listDistrib        
        
        #lancement du formatage et de la recuperation
        #des valeurs contenues dans les variables globales
        try:
            self.stringxbn = self.getbnInfos()
        except:
            self.stringxbn = "<?xml version=\"1.0\"?>\n"
            self.stringxbn += "<ANALYSISNOTEBOOK NAME=\"Notebook."
            self.stringxbn +=  "bndefault\" ROOT=\""
            self.stringxbn +=  "bndefault\">\n  <BNMODEL NAME=\""
            self.stringxbn +=  "bndefault\">"
        
        self.getListStatProp()

        try:
            self.stringxbn += self.getListDynProp()
        except:
            self.stringxbn += "      <DYNAMICPROPERTIES/>\n"
        
        try:
            self.stringxbn += self.getListVar()
        except:
            self.stringxbn += "      <VARIABLES/>\n"
            
        if (len(self.listStruct) != 0):
            self.getListStruct()
        else:
            self.stringxbn += "      <STRUCTURE/>\n"
            
        if (len(self.listDistrib) != 0):
            self.getListDistrib()
        else:
            self.stringxbn += "      <DISTRIBUTIONS/>\n"
            self.stringxbn += "      </BNMODEL>\n"
            self.stringxbn += "    </ANALYSISNOTEBOOK>"
            
        f.write(self.stringxbn)
        f.write('\n')
        
        f.close()
    
    def getbnInfos(self):
        bn = "<?xml version=\"1.0\"?>\n"
        bn += "<ANALYSISNOTEBOOK NAME=\"Notebook."
        bn +=  self.bnInfos.root + "\" ROOT=\"%s"
        bn +=  self.bnInfos.root + "\">\n  <BNMODEL NAME=\""
        bn +=  self.bnInfos.root + "\">"
        
        return bn
        
    def getListStatProp(self):
        self.stringxbn += "<STATICPROPERTIES><FORMAT>MSR DTAS XML</FORMAT>\n"
        self.stringxbn += "        <VERSION>1.0</VERSION>\n"
        self.stringxbn += "        <CREATOR>Open Bayes GUI</CREATOR>\n"
        self.stringxbn += "        </STATICPROPERTIES>\n"
    
    def getListDynProp(self):
        dp = "      <DYNAMICPROPERTIES>\n"
 
        for pt in listDynProp.dynPropType:
            dp += "        <PROPERTYTYPE NAME=\""
            dp += pt['NAME'] + "\""
            dp += " TYPE=\"" + pt['TYPE'] + "\""
            
            if (pt.has_key('ENUMSET')):
                dp += " ENUMSET=\"" + pt['ENUMSET'] + "\""
            
            
            if (pt.has_key('COMMENT')):
                dp += "><COMMENT>" + pt['COMMENT']
                dp += "</COMMENT>\n"
                dp += "          </PROPERTYTYPE>\n"
            else:
                dp += "/>\n"
    
        for name, value in listDynProp.dynProperty.items():
            dp += "          <PROPERTY NAME=\"" + name + "\">"
            dp += "<PROPVALUE>" + value + "</PROPVALUE>\n"
            dp += "            </PROPERTY>\n"
            
        for name, value in listDynProp.dynPropXml.items():
            dp += "          <PROPERTY NAME=\"" + name + "\">"
            dp += "<PROPVALUE>" + value + "</PROPVALUE>\n"
            dp += "            </PROPERTY>\n"
                
        dp += "        </DYNAMICPROPERTIES>\n"
        
        return dp
    
    def getListVar(self):
        var = "      <VARIABLES>\n"
        for info in self.listVar:
            var += "        <VAR NAME=\"" + info.name + "\""
            var += " TYPE=\"" + info.type + "\""
            var += " XPOS=\"" + info.xpos + "\""
            var += " YPOS=\"" + info.ypos + "\">"
            var += "<FULLNAME>" + info.description
            var += "</FULLNAME>\n"
 
            for s in info.stateName:
                var += "          <STATENAME>" + s
                var += "</STATENAME>\n"
            
            for name, value in info.propertyNameValue.items():
                var += "          <PROPERTY NAME=\"" + name + "\">"
                var += "<PROPVALUE>" + value + "</PROPVALUE>\n"
                var += "            </PROPERTY>\n"
            
            var += "          </VAR>\n"
        
        var += "        </VARIABLES>\n"
        
        return var
    
    def getListStruct(self):
        self.stringxbn += "      <STRUCTURE>\n"
        child = False
        
        for info in self.listStruct:
            if not child:
                self.stringxbn += "        <ARC PARENT=\"" + info + "\""
                child = True
            else:
                self.stringxbn += " CHILD=\"" + info + "\"/>\n"
                child = False
        
        self.stringxbn += "        </STRUCTURE>\n"
        
    def getListDistrib(self):
        self.stringxbn += "      <DISTRIBUTIONS>\n"
        
        for info in self.listDistrib:
            self.stringxbn += "        <DIST TYPE=\"" + info.type + "\">\n"
            
            if (len(info.condelem) > 0):
                self.stringxbn += "          <CONDSET>\n"
                
                for condelem in info.condelem:
                    self.stringxbn += "            <CONDELEM NAME=\""
                    self.stringxbn += condelem + "\"/>\n"
                self.stringxbn += "            </CONDSET>\n"
            
            self.stringxbn += "          <PRIVATE NAME=\""
            self.stringxbn += info.name + "\"/>\n"

            self.stringxbn += "          <DPIS>\n"
            x=0
            for data in info.dpiData:
                try:
                    if (len(info.condelem) > 0):
                        self.stringxbn += "            <DPI INDEXES=\""
                        self.stringxbn += info.dpiIndex[x] + "\">"
                    else:
                        self.stringxbn += "            <DPI>"

                    x += 1
                except:
                    pass
                self.stringxbn += str(data) + "</DPI>\n"
            self.stringxbn += "            </DPIS>\n"
            self.stringxbn += "          </DIST>\n"
        self.stringxbn += "        </DISTRIBUTIONS>\n"
        self.stringxbn += "      </BNMODEL>\n"
        self.stringxbn += "    </ANALYSISNOTEBOOK>"
        
        
#=======================================================================
#--- TESTS
#=======================================================================
class XBNTestCase(unittest.TestCase):   
         
    def setUp(self):
        """ reads the WetGrass.xbn """
        file_name_in = './WetGrass.xbn'
        xbn = LoadXBN(file_name_in)
        self.G = xbn.Load()


    def testGeneral(self):
        ' tests general data'
        assert(self.G.name == 'bndefault'), \
                " General BN data is not read correctly"
                
    def testDistributions(self):
        ' test the distributions'
        r = self.G.v['Rain']
        s = self.G.v['Sprinkler']
        w = self.G.v['Watson']
        h = self.G.v['Holmes']
        
        assert(allclose(r.distribution.cpt, [0.2, 0.8]) and \
                allclose(s.distribution.cpt, [0.1, 0.9]) and \
                allclose(w.distribution[{'Rain':1}], [0.2, 0.8])), \
                " Distribution values are not correct"

    def testInference(self):
        """ Loads the RainWatson BN, performs inference and checks the results """
        self.engine = JoinTree(self.G)
        r = self.engine.marginalise('Rain')        
        s = self.engine.marginalise('Sprinkler')
        w = self.engine.marginalise('Watson')
        h = self.engine.marginalise('Holmes')

        assert(allclose(r.cpt,[0.2,0.8]) and \
                allclose(s.cpt,[0.1,0.9]) and \
                allclose(w.cpt,[0.36,0.64]) and \
                allclose(h.cpt,[ 0.272, 0.728]) ), \
               " Somethings wrong with JoinTree inference engine"
            
    def testSave(self):
        ''' reads an xbn, then writes it again, then reads it again and then 
        perform inference and checks the results'''
        file_name_out = './test.xbn'
        SaveXBN(file_name_out, self.G)
        xbn = LoadXBN(file_name_out)
        self.G = xbn.Load()
        
        self.testInference()
        
        import os
        # delete the test.xbn file
        os.remove(file_name_out)


#---MAIN   
if __name__ == '__main__':
    
    suite = unittest.makeSuite(XBNTestCase, 'test')
    runner = unittest.TextTestRunner()
    runner.run(suite)
  

