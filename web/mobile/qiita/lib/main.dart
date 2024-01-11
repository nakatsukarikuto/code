import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:qiita/screens/search_screens.dart';

Future<void> main() async {
  await dotenv.load(fileName: '.env');
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        title: 'Qiita Search',
        theme: ThemeData(
          primarySwatch: Colors.green,
          fontFamily: 'Heragino Sans',
          appBarTheme: const AppBarTheme(
            backgroundColor: Color(0xFF55C500),
          ),
          textTheme: Theme.of(context).textTheme.apply(
                bodyColor: Colors.white,
              ),
        ),
        home: const SearchScreen());
  }
}
