'''
Simple HTTP server for the spacetree and parameter list utilities.
'''
import httplib2
import os
import simplejson
import urllib

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from cgi import FieldStorage
from datetime import datetime
from jinja2 import Environment, PackageLoader
from tempfile import mkstemp
from urlparse import urlparse

from hdfaccess.file import hdf_file

from analysis_engine import settings
from analysis_engine.dependency_graph import (
    graph_adjacencies,
    graph_nodes,
    process_order,
)
from analysis_engine.node import NodeManager
from analysis_engine.process_flight import get_derived_nodes
#from analysis_engine.settings import BASE_URL


# Currently only test implements parameter request.
BASE_URL = 'https://polaris-test.flightdataservices.com'


class GetHandler(BaseHTTPRequestHandler):
    _template_env = Environment(loader=PackageLoader('spacetree', 'templates'))
    
    # Helper methods
    ############################################################################
    
    def _respond(self, body, status=200, content_type='text/html'):
        '''
        Respond with body setting status and Content-type.
        '''
        self.send_response(status)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(body)
    
    def _respond_with_template(self, template_path, context={}, **kwargs):
        '''
        Respond with rendered template.
        '''
        template = self._template_env.get_template(template_path)
        self._respond(template.render(**context), **kwargs)
    
    def _respond_with_error(self, status, message):
        '''
        Respond with error status code and message.
        '''
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.send_error(status, message)
    
    # Method handlers.
    ############################################################################
    
    def do_POST(self):
        if self.path.endswith('/spacetree'):
            self._spacetree()
            return
        
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        #get_query = parse_qs(parsed_url.query)
        
        ##message_parts = [
                ##'CLIENT VALUES:',
                ##'client_address=%s (%s)' % (self.client_address,
                                            ##self.address_string()),
                ##'command=%s' % self.command,
                ##'path=%s' % self.path,
                ##'real path=%s' % parsed_path.path,
                ##'query=%s' % parsed_path.query,
                ##'request_version=%s' % self.request_version,
                ##'',
                ##'SERVER VALUES:',
                ##'server_version=%s' % self.server_version,
                ##'sys_version=%s' % self.sys_version,
                ##'protocol_version=%s' % self.protocol_version,
                ##'',
                ##'HEADERS RECEIVED:',
                ##]
        ##for name, value in sorted(self.headers.items()):
            ##message_parts.append('%s=%s' % (name, value.rstrip()))
        ##message_parts.append('')
        ##message = '\r\n'.join(message_parts)
        if path.endswith('/index'):
            self._index() # file upload page with aircraft info
            return
        elif path.endswith('/parameters'):
            self._parameters()
            return
        elif path.endswith('/spacetree'):
            self._index() # Without an HDF file in POST, redirect to index.
            return
        elif path.startswith('/_assets'):
            try:
                self._respond_with_static(path)
                return
            except IOError:
                pass
        self._respond_with_error(404, 'Page Not Found %s' % self.path)          
    
    # Static resources
    ############################################################################
    
    def _respond_with_static(self, path):
        file_path = path.lstrip('/')
        # try and serve from the current directory
        if file_path.endswith('.js'):
            content_type = 'text/javascript'
        elif file_path.endswith('.css'):
            content_type = 'text/css'
        else:
            content_type = 'text/html'
        self._respond(open(file_path).read(), 200, content_type)    
    
    # Page response methods
    ############################################################################
    
    def _index(self, error=None):
        '''
        :param error: Optional error to display with the form.
        :type error: str
        '''
        self._respond_with_template('index.html', {
            'error': error,
        })
    
    def _spacetree(self):
        form = FieldStorage(self.rfile, headers=self.headers,
                            environ={'REQUEST_METHOD': 'POST'})
        file_upload = form['hdf_file']
        if not file_upload.filename:
            self._index(error="Please select a file to upload.")
            return
        # Create and write to a temp file.
        file_desc, file_path = mkstemp()
        file_obj = os.fdopen(file_desc, 'w')
        file_obj.write(file_upload.file.read())
        file_obj.close()
        try:
            with hdf_file(file_path) as hdf_file_obj:
                lfl_params = hdf_file_obj.keys()
        except IOError:
            self._index(error="Please select a valid HDF file.")
            return        
        param_names = self._generate_json(lfl_params)
        polaris_query, params, missing_params = self._fetch_params(param_names)     
        self._respond_with_template('spacetree.html', {
            'missing_params': missing_params,
            'params': params,
            'polaris_query': polaris_query,
        })
        return
        
    
    # Generate data from HDF
    ############################################################################
    
    def _generate_json(self, lfl_params):
        # Ensure file is a valid hdf5 file before continuing.
        derived_nodes = get_derived_nodes(settings.NODE_MODULES)
        required_params = derived_nodes.keys()
        
        # TODO: Update ac_info with keys from provided fields
        ac_info = {'Tail Number': 'G-ABCD',
                   'Family': u'B737 NG',
                   'Series': u'B737-800',
                   'Main Gear To Lowest Point Of Tail': None,
                   'Manufacturer Serial Number': u'39009', 
                   'Main Gear To Radio Altimeter Antenna': None,
                   'Precise Positioning': True, 
                   'Model': u'B737-8JP', 
                   'Identifier': u'15', 
                   'Frame': u'737-3C',
                   'Manufacturer': u'Boeing'}
        # TODO: Option to populate
        achieved_flight_record = {}
        
        # generate dep tree
        node_mgr = NodeManager(datetime.now(),
                               lfl_params, 
                               required_params, 
                               derived_nodes,
                               ac_info,
                               achieved_flight_record)
        _graph = graph_nodes(node_mgr)
        gr_all, gr_st, order = process_order(_graph, node_mgr)

        # save to tree.json
        simplejson.dump(graph_adjacencies(gr_st), open('_assets/ajax/tree.json', 'w'), indent=2)
            
        # save nodes to node_list.json
        param_names = gr_st.nodes()
        simplejson.dump(param_names, open('_assets/ajax/node_list.json', 'w'), indent=2)
        return param_names
    
    # REST
    ############################################################################
    
    def _fetch_params(self, param_names):
        '''
        Fetch params from server.
        '''
        http = httplib2.Http(disable_ssl_certificate_validation=True)
        body = urllib.urlencode({'parameters': simplejson.dumps(param_names)})
        try:
            print 'Fetching parameters from: %s' % BASE_URL
            response, content = http.request(BASE_URL + '/api/parameter',
                                             'POST', body)
        except Exception as err:
            print 'Exception raised when querying website:', str(err)
            polaris_query = False
            params = {param_name: {'database': None,
                                   'limits': None} for param_name in param_names}
        else:
            polaris_query = True
            print 'content', content
            params = simplejson.loads(content)['data']
        mandatory_params = open('mandatory_params', 'r').read().splitlines()
        for param_name, param_info in params.iteritems():
            param_info['mandatory'] = param_name in mandatory_params
        missing_params = set(mandatory_params) - set(param_names)
        return polaris_query, params, missing_params
        

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8080), GetHandler)
    print 'Starting server at http://localhost:8080/index (use <Ctrl-C> to stop)'
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print '<Ctrl-C>  received, shutting down server'  
        server.socket.close()  
