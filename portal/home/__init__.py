from flask import Blueprint

from portal.home.home import (
    main, change_source, video_feed, key_press, run, stop, connect_controller,
    get_esp32_ports, connect_esp32, disconnect_esp32, get_esp32_status,
    flash_esp32, get_flash_status
)

home = Blueprint('home', __name__, template_folder='templates', url_prefix='/home')
home.route('/')(main)
home.route('/source', methods=['POST'])(change_source)
home.route('/video_feed')(video_feed)
home.route('/keypress', methods=['POST'])(key_press)
home.route('/run', methods=['POST'])(run)
home.route('/stop', methods=['POST'])(stop)
home.route('/connect_controller', methods=['POST'])(connect_controller)

# ESP32 相关路由
home.route('/esp32/ports', methods=['GET'])(get_esp32_ports)
home.route('/esp32/connect', methods=['POST'])(connect_esp32)
home.route('/esp32/disconnect', methods=['POST'])(disconnect_esp32)
home.route('/esp32/status', methods=['GET'])(get_esp32_status)
home.route('/esp32/flash', methods=['POST'])(flash_esp32)
home.route('/esp32/flash/status', methods=['GET'])(get_flash_status)
