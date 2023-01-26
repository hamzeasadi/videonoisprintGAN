import cv2
import os
import conf as cfg



def iframeextractionvideo(videopath, trgpath):
    # command = f"ffmpeg -skip_frame nokey -i {videopath} -vsync 0 -frame_pts true {filepath}out%d.png"
    nn = cfg.paths['data']
    srcfolders = os.listdir(videopath)
    srcfolders = cfg.ds_rm(srcfolders)
    for srcfolder in srcfolders:
        trgfolder = os.path.join(trgpath, srcfolder)
        srcfolderpath = os.path.join(videopath, srcfolder)
        cfg.creatdir(trgfolder)
        videos = os.listdir(srcfolderpath)
        videos = cfg.ds_rm(videos)
        for i, video in enumerate(videos):
            videopathfile = os.path.join(srcfolderpath, video)
            command = f"ffmpeg -skip_frame nokey -i {videopathfile} -vsync vfr -frame_pts true -x264opts no-deblock {trgfolder}/video{i}out%d.bmp"
            os.system(command=command)

def createpatches(srcpath, trgpath):
    srciframefolders = os.listdir(srcpath)
    srciframefolders = cfg.ds_rm(srciframefolders)
    for srciframefolder in srciframefolders:
        srciframefolderpath = os.path.join(srcpath, srciframefolder)
        srciframes = os.listdir(srciframefolderpath)
        srciframes = cfg.ds_rm(srciframes)
        for iframe in srciframes:
            iframepath = os.path.join(srciframefolderpath, iframe)
            img = cv2.imread(iframepath)
            h, w, c = img.shape
            hc, wc = h//2, w//2
            H, W = 720, 1280
            if h>w:
                os.remove(iframepath)
            else:
                newimg = img[hc-H//2:hc+H//2, wc-W//2:wc+W//2, :]
                cv2.imwrite(filename=iframepath, img=newimg)







def main():
    # srcpath = cfg.paths['videos']
    # trgpath = cfg.paths['iframes']
    # iframeextractionvideo(videopath=srcpath, trgpath=trgpath)

    srcpath = cfg.paths['iframes']
    trgpath = cfg.paths['patches']
    createpatches(srcpath=srcpath, trgpath=trgpath)

if __name__ == '__main__':
    main()