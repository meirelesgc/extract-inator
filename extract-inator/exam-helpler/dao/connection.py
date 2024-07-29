import psycopg2


def execScript_db(sql):
    con = conecta_db()
    cur = con.cursor()
    try:
        cur.execute(sql)
        con.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        con.rollback()
        cur.close()
        return 1
    cur.close()


def consultar_db(sql):
    try:
        con = conecta_db()
        cur = con.cursor()
        cur.execute(sql)
        recset = cur.fetchall()
        registros = []
        for rec in recset:
            registros.append(rec)
        con.close()
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        con.rollback()
        cur.close()
        return 1
    return registros


def conecta_db():
    password = "root"
    host = "localhost"

    con = psycopg2.connect(
        host=host,
        database="TNPGE",
        user="postgres",
        password=password,
    )
    return con
