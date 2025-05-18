import os
import sys
import paramiko
import time
import socket
import subprocess
from multiprocessing import Process, Event, Array, RLock, Value
import json


def sftp_connect(ip, username, private_key, port):
    transport = paramiko.Transport((ip, port))
    transport.connect(username=username, pkey=private_key)

    sftp = paramiko.SFTPClient.from_transport(transport)

    sftp.sock.settimeout(5)

    return sftp


def fd_of_files_local(filename, flag = "rb", offset=0):
    fd = open(filename, flag)
    fd.seek(offset, 1)

    return fd


def fd_of_files_remote(filename, sftp, flag = "rb", offset=0, pl=True):
    fd_rem = sftp.file(filename, flag)
    fd_rem.set_pipelined(pl)
    fd_rem.seek(offset, 1)

    return fd_rem


def close_all_files(fd):
    for fd_i in fd:
        for j in fd_i:
            j.close()


def sftp_up_open_foo(name_up, server_up):
    sftp_up = [[sftp_connect(server_up[i]['ip'], server_up[i]['username'], server_up[i]['key'], server_up[i]['port']) \
            for j in range(len(name_up[i]))] for i in range(len(name_up))]

    return sftp_up


def sftp_down_open_foo(server_down):
    sftp_down = []
    for serv in server_down:
        sftp_down.append(sftp_connect(serv['ip'], serv['username'], serv['key'], serv['port']))

    return sftp_down


def sftp_open_foo(name_up, server_up, server_down):
    sftp_up = sftp_up_open_foo(name_up, server_up)

    sftp_down = sftp_down_open_foo(server_down) 

    return sftp_up, sftp_down


def fd_up_open_foo(sftp_up, name_up):
    fd_read_up = [[0 for j in range(len(name_up[i]))] for i in range(len(name_up))]
    fd_write_up = [[0 for j in range(len(name_up[i]))] for i in range(len(name_up))]

    for i in range(len(name_up)):
        for j in range(len(name_up[i])):
            try:
                count_size_up = sftp_up[i][j].stat(os.path.basename(name_up[i][j])).st_size
            except Exception:
                count_size_up = 0

            fd_read_up[i][j] = fd_of_files_local(name_up[i][j], "rb", count_size_up)

            fd_write_up[i][j] = fd_of_files_remote(os.path.basename(name_up[i][j]), sftp_up[i][j], "ab")

    return fd_read_up, fd_write_up


def fd_down_open_foo(sftp_down, name_down):
    fd_read_down = [[0 for j in range(len(name_down[i]))] for i in range(len(name_down))]
    fd_write_down = [[0 for j in range(len(name_down[i]))] for i in range(len(name_down))]

    for i in range(len(name_down)):
        for j in range(len(name_down[i])):
            vr_path = os.path.join(os.path.getcwd(), 'data', name_down[i][j])

            if os.path.exists(vr_path):
                count_size_down = os.path.getsize(vr_path)
            else:
                count_size_down = 0

            fd_read_down[i][j] = fd_of_files_remote(name_down[i][j], sftp_down[i], "rb", count_size_down, False)

            fd_write_down[i][j] = fd_of_files_local(os.path.join(os.path.getcwd(), 'data', name_down[i][j]), "ab")

    return fd_read_down, fd_write_down


def fd_open_foo(sftp_up, sftp_down, name_up, name_down):
    fd_read_up, fd_write_up = fd_up_open_foo(sftp_up, name_up)
    fd_read_down, fd_write_down = fd_down_open_foo(sftp_down, name_down)

    return fd_read_up, fd_write_up, fd_read_down, fd_write_down


def open_foo(name_up, server_up, name_down, server_down):
    sftp_up, sftp_down = sftp_open_foo(name_up, server_up, server_down)

    fd_read_up, fd_write_up, fd_read_down, fd_write_down = fd_open_foo(sftp_up, sftp_down, name_up, name_down)

    return sftp_up, sftp_down, fd_read_up, fd_write_up, fd_read_down, fd_write_down


def close_foo(sftp_up, sftp_down, fd_read_up, fd_write_up, fd_read_down, fd_write_down):
    close_all_files(fd_read_up)
    close_all_files(fd_write_up)
    close_all_files(fd_read_down)
    close_all_files(fd_write_down)

    for i in sftp_up:
        for j in i:
            j.close()

    for i in sftp_down:
        i.close()


def foo_of_num():
    return 0


def file_transport(name_up, priority_up, server_up, name_down, priority_down, server_down, flag, lock, window_size, \
                   speed_up, speed_down):
    try:
        mask_up = [[1 for j in range(len(name_up[i]))] for i in range(len(name_up))]
        mask_down = [[1 for j in range(len(name_down[i]))] for i in range(len(name_down))]

        n = sum([len(mask_up[i]) for i in range(len(mask_up))] + [len(mask_down[i]) for i in range(len(mask_down))])

        sftp_up, sftp_down, fd_read_up, fd_write_up, fd_read_down, fd_write_down = open_foo(name_up, server_up, \
                                                                                            name_down, server_down)

        inflag = False
        chsz = 2 ** 15

        tsleep = foo_of_num()

        cchunk_up = 0
        cchunk_down = 0
        window_size_dop = 1
        count_packet = 0
        time_up = 0
        time_down = 0

        while n:
            if inflag:
                break

            for i in range(len(mask_up)):
                if inflag:
                    break

                for j in range(len(mask_up[i])):
                    if inflag:
                        break

                    if mask_up[i][j]:
                        for k in range(priority_up[i][j]):
                            vr_time = time.time()
                            count_packet += 1
                            cchunk_up += 1

                            if flag.value:
                                inflag = True
                                break
                            
                            chunk = fd_read_up[i][j].read(chsz)

                            if not chunk:
                                mask_up[i][j] = False

                                n -= 1
                                break
                            
                            fd_write_up[i][j].write(chunk)
                             
                            time_up += time.time() - vr_time

                            if time_up > 10:
                                lock.acquire()
                                speed_up.value = cchunk_up / time_up
                                speed_up.value *= (2 ** 15 * 8 / 1000000)
                                lock.release()
                                
                                time_up = 0
                                cchunk_up = 0
                           
                            if (count_packet > window_size.value):
                                count_packet = 0

                                close_all_files(fd_read_up)
                                close_all_files(fd_write_up)

                                fd_read_up, fd_write_up = fd_up_open_foo(sftp_up, name_up)

                                window_size.value += window_size_dop
                                if window_size.value > 65500:
                                    window_size_dop = 0
                                    window_size.value = 65500

                                window_size_dop *= 2

                        if tsleep:
                            time.sleep(tsleep)


            for i in range(len(mask_down)):
                if inflag:
                    break

                for j in range(len(mask_down[i])):
                    if inflag:
                        break

                    if mask_down[i][j]:
                        for k in range(priority_down[i][j]):
                            vr_time = time.time()
                            cchunk_down += 1

                            if flag.value:
                                inflag = True
                                break

                            chunk = fd_read_down[i][j].read(chsz)

                            if not chunk:
                                mask_down[i][j] = False

                                n -= 1
                                break
                            
                            fd_write_down[i][j].write(chunk)

                            time_down += time.time() - vr_time

                            if time_down > 10:
                                lock.acquire()
                                speed_down.value = cchunk_down / time_down
                                speed_down.value *= (2 ** 15 * 8 / 1000000)
                                lock.release()
                                
                                time_down = 0
                                cchunk_down = 0


                        if tsleep:
                            time.sleep(tsleep)


        close_foo(sftp_up, sftp_down, fd_read_up, fd_write_up, fd_read_down, fd_write_down)

        lock.acquire()
        flag.value = 2
        lock.release()
    except Exception:
        return
