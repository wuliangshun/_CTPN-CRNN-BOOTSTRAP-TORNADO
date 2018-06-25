import tornado.ioloop
import tornado.web
import os
import ScenceRecognition_master as srm



from tornado.options import define, options
define("port", default=8888, help="run on the given port", type=int)

submit_num = 0
#服务器配置信息
settings = {
    "static_path": os.path.join(os.path.dirname(__file__), "static") ,
    "template_path": os.path.join(os.path.dirname(__file__), "templates"),
    "debug":True,
}





class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", sim_preds="", file_name="", display="none")
        
    def post(self, *args, **kwargs):
        #上传文件
        global submit_num
        submit_num = submit_num + 1
        if submit_num > 1:
            first = False
        else:
            first = True
        file_metas = self.request.files["fff"]        
        #for meta in file_metas:
        meta = file_metas[0]
        file_name = meta['filename']
        img_path = os.path.join(os.path.dirname(__file__),'ScenceRecognition_master','imgs',file_name)
        with open(img_path,'wb') as up:
            up.write(meta['body'])
        #调用定位识别
        raw_preds,sim_preds = srm.call_demo.demo(file_name, first = first)       
        #跳转
        new_file_name = '.'.join(file_name.split('.')[:-1])+'ctpn.' + file_name.split('.')[-1]
        self.render("index.html",  sim_preds=sim_preds, file_name=new_file_name, display="block")
        #self.redirect("/") 





      
        

if __name__ == "__main__":

    application = tornado.web.Application([(r"/", IndexHandler), 
                                        ( r'/(.*)', tornado.web.StaticFileHandler, {'path':os.path.dirname(__file__)} ), ], **settings)
    
    print(u'监听端口:%d\r\n' % options.port)
    
    application.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()