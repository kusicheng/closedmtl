import os
import zipfile
import tempfile
import shutil

def read_img(zip_path:str="nothing"):
    if not os.path.exists(zip_path):
        return ["ERR_PATHDNE"]
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            infos = zip_ref.infolist()
            total_size = sum(info.file_size for info in infos)
            
            #zip bomb check
            if total_size > 100 * 1024 * 1024:
                os.remove(zip_path)
                return ["ERR_MAL_SIZ"]
            
            #check for malicious extensions
            forbidden_exts = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.vbs', '.js', '.msi'] 
            for info in infos:
                _, ext = os.path.splitext(info.filename)
                if ext.lower() in forbidden_exts:
                    os.remove(zip_path)
                    return ["ERR_MAL_EXT"]
            
            #safety net to prevent small file spam
            if len(infos) > 1000: 
                os.remove(zip_path)
                return ["ERR_MAL_CNT"]
            
            #extract img
            allowed_exts = ['.png', '.jpg', '.jpeg']
            images_temp = tempfile.mkdtemp()
            image_paths = []
            
            for info in infos:
                if not info.is_dir():
                    _, ext = os.path.splitext(info.filename)
                    if ext.lower() in allowed_exts:
                        filename = os.path.basename(info.filename)
                        dest_path = os.path.join(images_temp, filename)
                        with open(dest_path, 'wb') as f:
                            f.write(zip_ref.read(info))
                        image_paths.append(dest_path)
            
            #note: images_temp is a persistent temp directory. caller is responsible for cleanup
            return image_paths
    
    except (zipfile.BadZipFile, Exception):
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return ["ERR_GEN"]